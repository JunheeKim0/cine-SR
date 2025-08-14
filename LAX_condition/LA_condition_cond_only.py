import os, sys
# 수정: 현재 파일의 상위 두 단계 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
import torch
# import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
from bitsandbytes.optim import AdamW8bit
import SimpleITK as sitk
from templates import *  # autoenc_base, LitModel, ModelName
import ants
from GAN import *

def main():
    # --- 1) Single-GPU setup --- 
    # 수정: distributed setup 제거하고 single GPU 사용
    # local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)
    device = torch.device("cuda:0")  # 수정: 첫 번째 GPU 사용
    # dist.init_process_group(backend="nccl")
    # world_size = dist.get_world_size()
    # assert world_size == 1, f"4 GPU 전용 스크립트입니다. 현재 world_size={world_size}"
    local_rank = 0   # 수정: 로컬 랭크 항상 0
    world_size = 1   # 수정: world_size 항상 1

    # 2) Model & EMA load
    conf = autoenc_base()
    conf.use_checkpoint = False
    conf.attn_checkpoint = False
    conf.img_size = 128
    conf.net_ch = 128
    conf.net_ch_mult = (1,1,2,3,4)
    conf.net_enc_channel_mult = (1,1,2,3,4,4)
    conf.model_name = ModelName.beatgans_autoenc
    conf.name = 'med256_autoenc'
    conf.make_model_conf()

    base_model = LitModel(conf)
    ckpt = torch.load(
        '/workspace/storage/kjh/cardiac/DMCVR/generation/diffae_multi_PNU/checkpoints_multi/med256_autoenc/1/epoch=438-step=756789.ckpt', map_location="cpu"
    )
    base_model.load_state_dict(ckpt["state_dict"], strict=True)
    base_model.to(device).eval()
    ema = base_model.ema_model.to(device).eval()
    model = ema

    # 3) Data preparation
    frame = 0  # if needed
    t =30
    num_epoch = 100
    # lambda_roi    = 5.0  
    # lambda_latent = 0.02   
    filling = 8
    lambda_roi    = 100.0  
    lambda_latent = 0.2   
    lambda_smooth = 0.2


    id_list = [
        148, 400, 769, 834, 243, 813, 600, 100, 708,
        342, 790, 823, 114, 760, 665, 812, 788, 515, 810,
        814, 786, 835, 182, 410, 724, 526, 704, 723, 387,
        776, 488, 711, 482, 190, 648, 692, 722, 664, 599
    ]

    # 2. 전처리 함수 (bias + clipping)
    def preprocess_n4_clip(arr, low_pct=1, high_pct=98):
        img_ants = ants.from_numpy(arr)
        arr_n4   = ants.n4_bias_field_correction(img_ants).numpy()
        lo, hi   = np.percentile(arr_n4, [low_pct, high_pct])
        return np.clip(arr_n4, lo, hi)/ hi
    
    def inter_sax_load(file_path) :
        img_itk_inter = sitk.ReadImage(file_path)
        vol_np_inter  = sitk.GetArrayFromImage(img_itk_inter)

        vol_np_inter = preprocess_n4_clip(vol_np_inter)
        vol_np_inter = vol_np_inter / np.max(vol_np_inter)

        print("flip!!!!!")
        vol_np_inter = np.flip(vol_np_inter, axis=-2).copy()

        # 빈 슬라이스(모두 0) 제거
        nonzero_idx = [i for i in range(len(vol_np_inter)) if vol_np_inter[i].max() != 0]
        vol_nz_inter      = vol_np_inter[nonzero_idx]
        return vol_nz_inter

    # 3. Quantile‐mapping 헬퍼
    def make_quantile_map(src_vals, tgt_vals, n_quant=101):
        qs    = np.linspace(0, 1, n_quant)
        src_q = np.quantile(src_vals, qs)
        tgt_q = np.quantile(tgt_vals, qs)
        return src_q, tgt_q

    # --- 1) Interpolation 함수 정의 ---
    def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
        omega = torch.acos((a * b).sum() / (a.norm() * b.norm()))
        return (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / torch.sin(omega)


    def projection_torch(vol: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        vol : (D, H, W)   – 3D volume tensor  
        mask: (D, H, W)   – 3D binary/probability mask tensor  
        returns: (H, W)  – XY-plane mean projection
        """

        # 1) ROI masking (element‑wise)
        masked_vol = vol * mask  # shape = (D, H, W)

        # 2) Z‑axis(depth) 방향으로 합(sum) 및 평균 계산
        sum_vol  = masked_vol.sum(dim=1)                 # (H, W)  
        sum_mask = mask.sum(dim=1).clamp(min=1e-6)       # (H, W)  
        mean_roi = sum_vol / sum_mask                    # (H, W)

        return mean_roi

    def load_lax_mask(lax_path, mask_path, device):
        gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(lax_path.format(frame=frame))).astype(np.float32)
        gt_vol = preprocess_n4_clip(gt_vol, low_pct=1, high_pct=99.5)

        gt_vol = np.flip(gt_vol, axis=1).copy()

        mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(mask_path.format(frame=frame))).astype(bool)
        mask_vol = np.flip(mask_vol, axis=1).copy()

        gt_vol = torch.from_numpy(gt_vol).float().to(device)
        mask_vol = torch.from_numpy(mask_vol).float().to(device)

        gt_vol_proj = projection_torch(gt_vol, mask_vol)

        return gt_vol_proj, mask_vol
    
    def affine_normalize(sax_img: torch.Tensor, lax_img: torch.Tensor):
        """
        img1의 intensity를 img2와 유사하게 affine normalization

        Args:
            img1: (H, W) – source image
            img2: (H, W) – reference image

        Returns:
            img1_normed: (H, W) – img2와 intensity 분포를 맞춘 img1
        """

        mean1 = sax_img.mean()
        std1  = sax_img.std().clamp(min=1e-6)

        mean2 = lax_img.mean()
        std2  = lax_img.std().clamp(min=1e-6)

        # Affine normalization
        sax_normed = (sax_img - mean1) / std1 * std2 + mean2
        return sax_normed


    for id in id_list : 
        # -- SAX volume
        sax_path = f"/workspace/storage/kjh/dataset/cardiac/PNU_cardiac/CINE/test/middle_slice/sax/MR_Heart_{id}_crop_sa.nii.gz"
        img_sax = sitk.ReadImage(sax_path)
        vol_sax = sitk.GetArrayFromImage(img_sax).astype(np.float32)  # (Z,H,W)
        vol_sax = preprocess_n4_clip(vol_sax)
        vol_sax = vol_sax * 2 - 1

        vol_sax = np.flip(vol_sax, axis=-2)  # flip y
        # remove zero slices
        nonzero = [i for i in range(vol_sax.shape[0]) if vol_sax[i].max() != 0]
        vol_sax = vol_sax[nonzero]

        vol_sax = vol_sax[frame, :, :, :]

        # ======================= LAX Load ================ #


        gt_2ch_path   = f"/workspace/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{id}/2ch_nii_flipped_aligned_1lax_cropped/2ch_flipped_aligned_1lax_cropped_{frame}.nii.gz"
        gt_3ch_path   = f"/workspace/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{id}/3ch_nii_flipped_aligned_1lax_cropped/3ch_flipped_aligned_1lax_cropped_{frame}.nii.gz"
        gt_4ch_path   = f"/workspace/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{id}/4ch_nii_aligned_1lax_cropped/4ch_aligned_1lax_cropped_{frame}.nii.gz"

        mask_2ch_path = f"/workspace/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{id}/2ch_nii_flipped_aligned_1lax_mask_cropped_adjusted/2ch_flipped_aligned_1lax_mask_cropped_adjusted_{frame}.nii.gz"
        mask_3ch_path = f"/workspace/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{id}/3ch_nii_flipped_aligned_1lax_mask_cropped_adjusted/3ch_flipped_aligned_1lax_mask_cropped_adjusted_{frame}.nii.gz"
        mask_4ch_path = f"/workspace/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{id}/4ch_nii_aligned_1lax_mask_cropped_adjusted/4ch_aligned_1lax_mask_cropped_adjusted_{frame}.nii.gz"

        proj_gt_2ch, mask_2ch = load_lax_mask(gt_2ch_path, mask_2ch_path, device)
        proj_gt_3ch, mask_3ch = load_lax_mask(gt_3ch_path, mask_3ch_path, device)
        proj_gt_4ch, mask_4ch = load_lax_mask(gt_4ch_path, mask_4ch_path, device)



        # convert to torch
        sax_tensor = torch.from_numpy(vol_sax).float().unsqueeze(1).to(device)    # (Z,1,H,W)

        # encode + stochastic encode
        cond = base_model.encode(sax_tensor)
        xTs = base_model.encode_stochastic(sax_tensor, cond, T=30)

        # --- 2) 시퀀스 생성 ---
        xs, cs = [], []
        N = xTs.shape[0]
        for i in range(N - 1):
            a, b = xTs[i], xTs[i + 1]
            ca, cb = cond[i], cond[i + 1]
            xs.append(a); cs.append(ca)
            for alpha in np.linspace(0, 1, filling, endpoint=False)[1:]:
                xs.append(slerp(a, b, float(alpha)))
                cs.append(ca * (1 - alpha) + cb * alpha)
        xs.append(xTs[-1]); cs.append(cond[-1])

        before_interp_xTs = torch.stack(xs, dim=0)
        before_interp_cs  = torch.stack(cs,  dim=0)

        # --- 3) GPU별 index 할당 ---
        total      = before_interp_xTs.shape[0]
        idxs  = list(range(total))

        # # --- 4) 최적화 대상 Parameter & 초기값 저장 ---
        # my_interp_xTs = torch.nn.Parameter(
        #     before_interp_xTs[idxs].clone().detach().to(device),
        #     requires_grad=True
        # )
        my_interp_xTs = before_interp_xTs[idxs].clone().detach().to(device)
        my_interp_xTs.requires_grad = True 

        my_interp_cs = torch.nn.Parameter(
            before_interp_cs[idxs].clone().detach().to(device),
            requires_grad=True
        )

        # 초기값 저장 (prior 계산용)
        my_interp_xTs_init = my_interp_xTs.data.clone()
        my_interp_cs_init  = my_interp_cs.data.clone()

        # --- 5) Optimizer, Scheduler, AMP 설정 ---
        optimizer = AdamW8bit([my_interp_cs], lr=1e-2, weight_decay=1e-2)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        scaler    = GradScaler()


        # --- 6) Training Loop (전체 loss) ---
        for epoch in range(num_epoch):
            optimizer.zero_grad()
            # ◆한 번에 전체 처리◆
            with autocast():
                pred_all = base_model.render(
                    my_interp_xTs,  # 전체 슬라이스
                    my_interp_cs,
                    t
                )
                samples = pred_all["sample"]         # (N,1,H,W)
                samples = (samples + 1) / 2          # normalization

            # 전체 볼륨 구성
            vol_pred = samples.squeeze(1)            # (N,H,W)

            # --- 6.1) 인접 슬라이스 Smoothness Regularization 손실 계산 ---
            # vol_pred[i] 와 vol_pred[i+1]의 차이가 크지 않도록 제약
            diffs = vol_pred[1:] - vol_pred[:-1]     # (N-1, H, W)
            smooth_loss = diffs.pow(2).mean()        # L2 penalty[^2]

            # SAX projection & affine normalization
            sax_proj_2ch = affine_normalize(projection_torch(vol_pred, mask_2ch), proj_gt_2ch)
            sax_proj_3ch = affine_normalize(projection_torch(vol_pred, mask_3ch), proj_gt_3ch)
            sax_proj_4ch = affine_normalize(projection_torch(vol_pred, mask_4ch), proj_gt_4ch)

            # ROI MSE loss
            loss_2ch = F.mse_loss(sax_proj_2ch, proj_gt_2ch, reduction='sum') / mask_2ch.sum().float()
            loss_3ch = F.mse_loss(sax_proj_3ch, proj_gt_3ch, reduction='sum') / mask_3ch.sum().float()
            loss_4ch = F.mse_loss(sax_proj_4ch, proj_gt_4ch, reduction='sum') / mask_4ch.sum().float()
            total_mse_loss = loss_2ch + loss_3ch + loss_4ch

            # Latent prior loss
            latent_reg = ((my_interp_xTs - my_interp_xTs_init)**2).mean()
            cond_reg   = ((my_interp_cs - my_interp_cs_init)**2).mean()
            prior_loss = lambda_latent * (latent_reg + cond_reg)

            # --- 3) Total loss & backward (Smoothness 포함) ---
            total_loss = (
                lambda_roi   * total_mse_loss +
                prior_loss +
                lambda_smooth * smooth_loss
            )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if local_rank == 0:
                print(f"[Epoch {epoch+1}/{num_epoch}] "
                    f"Loss={total_loss.item():.6f}  "
                    f"Smooth={smooth_loss.item():.6f}")  # smooth loss 모니터링

        print("Optimization 완료")

        # === 최적화된 slice 저장 ===

        # 메타 추출
        sid = os.path.basename(sax_path).split("_")[2]  # e.g. "834"
        frame_idx = frame

        orig_spacing_4d   = img_sax.GetSpacing()    # (sx, sy, sz, st)
        orig_origin_4d    = img_sax.GetOrigin()     # (ox, oy, oz, ot)
        orig_direction_4d = img_sax.GetDirection()  # len=16

        # --- 3D용으로 슬라이스 메타 구성 (변경 없음) ---
        spacing3d  = orig_spacing_4d[:3]
        origin3d   = orig_origin_4d[:3]
        if len(orig_direction_4d) == 16:
            direction3d = (
                orig_direction_4d[0], orig_direction_4d[1], orig_direction_4d[2],
                orig_direction_4d[4], orig_direction_4d[5], orig_direction_4d[6],
                orig_direction_4d[8], orig_direction_4d[9], orig_direction_4d[10],
            )
        else:
            direction3d = orig_direction_4d

        orig_n = vol_sax.shape[0]

        opt_xTs = my_interp_xTs.detach()        # 수정: 바로 전체 사용
        opt_cs  = my_interp_cs.detach()         # 수정: 바로 전체 사용

        # --- 모델로 렌더링 후 정규화 ---
        with torch.no_grad():
            out = base_model.render(opt_xTs, opt_cs, t)  # (N,1,H,W)
            pred_after = (out["sample"] + 1) / 2         # 수정: [-1,1]→[0,1]

            before = base_model.render(before_interp_xTs, before_interp_cs)
            pred_before = (before["sample"] + 1) / 2

        # --- NumPy 배열로 변환 및 flip 처리 ---
        vol_after = pred_after.squeeze(1).cpu().numpy().astype(np.float32)
        vol_after = np.flip(vol_after, axis=1).copy()    # 수정: y축 뒤집기

        vol_pre = pred_before.squeeze(1).cpu().numpy().astype(np.float32)
        vol_pre = np.flip(vol_pre, axis=1).copy()

        # --- z 간격 보정 ---
        new_n = vol_after.shape[0]
        new_z = spacing3d[2] * (orig_n - 1) / (new_n - 1) 

        # --- 원본 SAX volume도 동일 spacing/new_z 로 저장 ---
        itk_orig = sitk.GetImageFromArray(vol_pre.astype(np.float32))
        itk_orig.SetSpacing((spacing3d[0], spacing3d[1], new_z))
        itk_orig.SetOrigin((origin3d[0], origin3d[1], origin3d[2]))
        itk_orig.SetDirection(direction3d)
        save_dir_o = f"/workspace/storage/kjh/dataset/cardiac/output/DiffAE/subject_{sid}/frame_{frame_idx}/"
        os.makedirs(save_dir_o, exist_ok=True)
        save_path_o = os.path.join(
            save_dir_o,
            f"MR_Heart_{sid}_frame_{frame_idx}_original_{orig_n}slice.nii.gz"
        )
        sitk.WriteImage(itk_orig, save_path_o)
        print(f"[Rank {local_rank}] Saved original SAX volume → {save_path_o}")

        # === 최적화된 slice 저장 ===
        itk_after = sitk.GetImageFromArray(vol_after)
        itk_after.SetSpacing((spacing3d[0], spacing3d[1], new_z))
        itk_after.SetOrigin((origin3d[0], origin3d[1], origin3d[2]))
        itk_after.SetDirection(direction3d)
        save_dir_a  = f"/workspace/storage/kjh/dataset/cardiac/output/Inverse/subject_{sid}/frame_{frame_idx}/"
        os.makedirs(save_dir_a, exist_ok=True)
        save_path_a = os.path.join(
            save_dir_a,
            f"MR_Heart_{sid}_frame_{frame_idx}_cond_only_{new_n}slice.nii.gz"
        )
        sitk.WriteImage(itk_after, save_path_a)
        print(f"[Rank {local_rank}] Saved post-opt volume → {save_path_a}")


if __name__=="__main__":
    # torchrun --nproc_per_node=4 LA_condition_all_pred_after.py
    main()