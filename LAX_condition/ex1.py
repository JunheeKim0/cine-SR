## slice 기반 Affine normalization ##
## slice similarity 유지 ( prior loss, slice similarity loss 등으로 최적화 완료 )


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
import torch
# import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
# from bitsandbytes.optim import AdamW8bit
from torch.optim import Adam
import SimpleITK as sitk
from templates import *  # autoenc_base, LitModel, ModelName
import ants
from GAN import *
from torch.amp import autocast 

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
    num_epoch = 30
    filling = 8
    # lambda_roi    = 3.0  
    # lambda_latent = 0.2  
    lambda_roi    = 5.0
    lambda_prior  = 1.0           # 초기 prior  가중치
    target_smooth_ratio = 0.01     # smooth_term / total_loss 의 목표 비율 (10%)
    target_prior_ratio  = 0.05    # prior_term  / total_loss 의 목표 비율 (5%)
    alpha_s = 0.5                 # smooth λ 조절 gain
    alpha_p = 0.1                 # prior  λ 조절 gain

    lambda_latent = 0.02
    alpha_proj    = 0.001
    lambda_smooth = 3.0  



    # 148 제외외
    id_list = [
        400, 769, 834, 243, 813, 600, 100, 708,
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
    
    def affine_normalize(sax_img: torch.Tensor, lax_mu, lax_sigma, n_valid):

        # 2) ROI 통계 및 Affine normalization
        mu_pred  = sax_img.sum() / n_valid
        sigma_pred = torch.sqrt(
            ((sax_img - mu_pred) * mask_vol).pow(2).sum() / n_valid + 1e-3
        )
        sigma_pred = torch.clamp(sigma_pred, min=1e-3)
        pred_norm     = (sax_img - mu_pred) * (lax_sigma / sigma_pred) + lax_mu
        
        return pred_norm

    def compute_sax_roi_stats(sax_roi: torch.Tensor, mask_vol: torch.Tensor, slice_n_valid, eps: float = 1e-6):
        """
        Compute per-slice ROI-based mean and standard deviation for LAX volume.

        Parameters:
            lax_vol (torch.Tensor): shape (N, H, W), LAX slices
            mask_vol (torch.Tensor): shape (N, H, W), binary ROI mask per slice (1 in ROI, 0 outside)
            eps (float): small value for numerical stability

        Returns:
            mu_gt (torch.Tensor): shape (N,), per-slice ROI mean
            sigma_gt (torch.Tensor): shape (N,), per-slice ROI standard deviation
        """

        # Compute ROI mean per slice
        mu_gt = (sax_roi * mask_vol).sum(dim=(1, 2)) / n_valid  # (N,)

        # Compute ROI variance per slice
        diff = (sax_roi - mu_gt[:, None, None]) * mask_vol      # (N, H, W)
        var_gt = (diff ** 2).sum(dim=(1, 2)) / slice_n_valid           # (N,)

        # Standard deviation
        sigma_gt = torch.sqrt(var_gt + eps)                     # (N,)

        return mu_gt, sigma_gt


    def compute_lax_roi_stats(lax_vol: torch.Tensor, mask_vol: torch.Tensor, eps: float = 1e-6):
        """
        Compute per-slice ROI-based mean and standard deviation for LAX volume.

        Parameters:
            lax_vol (torch.Tensor): shape (N, H, W), LAX slices
            mask_vol (torch.Tensor): shape (N, H, W), binary ROI mask per slice (1 in ROI, 0 outside)
            eps (float): small value for numerical stability

        Returns:
            mu_gt (torch.Tensor): shape (N,), per-slice ROI mean
            sigma_gt (torch.Tensor): shape (N,), per-slice ROI standard deviation
        """
        # Number of valid ROI pixels per slice
        n_valid = mask_vol.sum(dim=(1, 2)).clamp(min=1)  # (N,)

        # Compute ROI mean per slice
        mu_gt = (lax_vol * mask_vol).sum(dim=(1, 2)) / n_valid  # (N,)

        # Compute ROI variance per slice
        diff = (lax_vol - mu_gt[:, None, None]) * mask_vol      # (N, H, W)
        var_gt = (diff ** 2).sum(dim=(1, 2)) / n_valid           # (N,)

        # Standard deviation
        sigma_gt = torch.sqrt(var_gt + eps)                     # (N,)

        return mu_gt, sigma_gt, n_valid

    def slicewise_affine_normalize(
        sax_vol: torch.Tensor,
        sax_mask: torch.Tensor,
        mu_sax: torch.Tensor,
        sigma_sax: torch.Tensor,
        mu_gt: torch.Tensor,
        sigma_gt: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Perform per-slice affine normalization of SAX volume to LAX statistics.

        Parameters:
            sax_vol    : (N, H, W) SAX volume tensor
            sax_mask   : (N, H, W) binary ROI mask for SAX slices (1 in ROI, 0 outside)
            mu_sax     : (N,) per-slice SAX ROI mean
            sigma_sax  : (N,) per-slice SAX ROI std-dev
            mu_gt      : (N,) per-slice LAX ROI mean
            sigma_gt   : (N,) per-slice LAX ROI std-dev
            eps        : small epsilon for numerical stability

        Returns:
            sax_norm   : (N, H, W) slice-wise normalized SAX volume (ROI only)
        """
        # reshape statistics for broadcast
        mu_sax_   = mu_sax   .view(-1, 1, 1)  # (N,1,1)
        sigma_sax_= sigma_sax.view(-1, 1, 1).clamp(min=eps)
        mu_gt_    = mu_gt    .view(-1, 1, 1)
        sigma_gt_ = sigma_gt .view(-1, 1, 1)

        # affine normalize each slice
        sax_norm = (sax_vol - mu_sax_) * (sigma_gt_ / sigma_sax_) + mu_gt_

        # keep only ROI region
        sax_norm = sax_norm * sax_mask

        return sax_norm


    for id in id_list : 
        sax_path = f"/workspace/storage/kjh/dataset/cardiac/PNU_cardiac/CINE/test/middle_slice/sax/MR_Heart_{id}_crop_sa.nii.gz"
        img_sax = sitk.ReadImage(sax_path)
        vol_sax = sitk.GetArrayFromImage(img_sax).astype(np.float32)  # (Z,H,W)
        vol_sax = preprocess_n4_clip(vol_sax)
        mx = np.percentile(vol_sax, 98)
        vol_sax = np.clip(vol_sax, 0, mx) / mx * 2.0 - 1.0
        # vol_sax = vol_sax / np.max(vol_sax) 
        # vol_sax = vol_sax * 2.0 - 1.0
        vol_sax = np.flip(vol_sax, axis=-2)  # flip y
        # remove zero slices
        nonzero = [i for i in range(vol_sax.shape[0]) if vol_sax[i].max() != 0]
        vol_sax = vol_sax[nonzero]

        vol_sax = vol_sax[frame, :, :, :]

        # -- GT volume (3D) and mask volume
        gt_path   = f"/workspace/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{id}/combine_1lax_cropped/combine_1lax_cropped_{frame}.nii.gz"
        mask_path = f"/workspace/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{id}/combine_1lax_mask_cropped_adjusted/combine_1lax_mask_cropped_adjusted_{frame}.nii.gz"
        # interp_sax_path = f"/workspace/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{id}/sax_resample_linear_cropped/sax_resample_linear_cropped_{frame}.nii.gz"

        print("sax path:", sax_path)
        print("gt_path:", gt_path)
        print("mask_path:", mask_path)

        gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(gt_path.format(frame=frame))).astype(np.float32)
        gt_vol = preprocess_n4_clip(gt_vol)
        # gt_vol = gt_vol / np.max(gt_vol)
        gt_vol = np.clip(gt_vol, 0, np.nanpercentile(gt_vol, 98)) / np.nanpercentile(gt_vol, 98)
        gt_vol = np.flip(gt_vol, axis=1).copy()

        mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(mask_path.format(frame=frame))).astype(bool)
        mask_vol = np.flip(mask_vol, axis=1).copy()

        # convert to torch
        sax_tensor = torch.from_numpy(vol_sax).float().unsqueeze(1).to(device)    # (Z,1,H,W)
        gt_tensor  = torch.from_numpy(gt_vol).float().unsqueeze(1).to(device)      # (Z,1,H,W)
        mask_tensor= torch.from_numpy(mask_vol).float().unsqueeze(1).to(device)    # (Z,1,H,W)

        # encode + stochastic encode
        cond = base_model.encode(sax_tensor)
        xTs = base_model.encode_stochastic(sax_tensor, cond, T=30)


        # --- 2) 시퀀스 생성 ---
        filling = 8
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
        num_per_gpu = total // world_size
        remainder   = total % world_size
        start = local_rank * num_per_gpu + min(local_rank, remainder)
        end   = start + num_per_gpu + (1 if local_rank < remainder else 0)
        idxs  = list(range(start, end))

        # --- 4) 최적화 대상 Parameter & 초기값 저장 ---
        # (기존과 동일하게 before_interp_xTs, before_interp_cs 로부터 batch-sized Parameter 생성)
        x = torch.nn.Parameter(before_interp_xTs[idxs].clone().detach().to(device), requires_grad=True)  # shape=(N,1,H,W)
        c = torch.nn.Parameter(before_interp_cs[idxs].clone().detach().to(device), requires_grad=True)   # shape=(N,cond_dim)

        # 초기값 저장 (prior 계산용)
        x0 = x.data.clone()
        c0 = c.data.clone()

        # --- 5) Optimizer, Scheduler, AMP 설정 ---
        optimizer = torch.optim.Adam([x, c], lr=2e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        scaler    = torch.cuda.amp.GradScaler()

        # GT/LAX 통계 계산 (한 번만)
        n_valid = mask_tensor.sum().float()
        gt_vol     = gt_tensor.permute(1,0,2,3)      # (1,N,H,W)
        mask_vol   = mask_tensor.permute(1,0,2,3)    # (1,N,H,W)
        mu_gt, sigma_gt, slice_n_valid = compute_lax_roi_stats(
            lax_vol=gt_vol.squeeze(0), mask_vol=mask_vol.squeeze(0)
        )
        gt_roi_vol = (gt_vol * mask_vol).squeeze(0)   # (N,H,W)

        # --- 추가) slice-wise 상태 변수 초기화 ---
        N = x.shape[0]
        best_loss         = torch.full((N,), float('inf'), device=device)
        epochs_no_improve = torch.zeros(N, dtype=torch.int, device=device)
        patience          = 10   # 개선이 없으면 해당 slice 최적화 중단
        active            = torch.ones(N, dtype=torch.bool, device=device)

        # --- gradient masking hook 등록 ---
        def mask_gradients(grad):
            grad = grad.clone()
            grad[~active] = 0
            return grad

        x.register_hook(mask_gradients)
        c.register_hook(mask_gradients)



        # --- 6) Training Loop (batch-wise) ---
        for epoch in range(num_epoch):
            with torch.cuda.amp.autocast():
                out      = base_model.render(x, cond=c, T=t)    # (N,1,H,W)
                samples  = (out["sample"] + 1) / 2             # (N,1,H,W)
                vol_pred = samples.squeeze(1)                  # (N,H,W)

                # 1) per‑slice affine normalization (기존)
                mu_sax, sigma_sax = compute_sax_roi_stats(vol_pred, mask_vol.squeeze(0), slice_n_valid)
                sax_norm = slicewise_affine_normalize(
                    vol_pred, mask_vol.squeeze(0),
                    mu_sax, sigma_sax, mu_gt, sigma_gt
                )                                              # (N,H,W)

                # 2) per‑slice ROI MSE (기존)
                diff           = (sax_norm - gt_roi_vol).pow(2) * mask_vol.squeeze(0)
                per_slice_loss = diff.sum(dim=(1,2)) / slice_n_valid  # (N,)
                roi_loss       = per_slice_loss.mean()

                # 3) 양측 smooth loss (Discrete Laplacian) 추가
                #    슬라이스 간 second derivative(이차 미분) 패널티
                center      = vol_pred[1:-1]       # 중간 슬라이스
                prev        = vol_pred[:-2]        # 바로 이전 슬라이스
                next_slice  = vol_pred[2:]         # 바로 다음 슬라이스
                laplacian   = 2*center - prev - next_slice
                smooth_loss = laplacian.pow(2).mean()  # scalar

                # 4) prior loss (기존)
                latent_reg = ((x - x0)**2).mean()
                cond_reg   = ((c - c0)**2).mean()
                prior_loss = lambda_latent * (latent_reg + cond_reg)

                # 5) total loss 에 smooth_loss 반영
                total_loss = (
                    lambda_roi    * roi_loss
                + lambda_prior * prior_loss
                + lambda_smooth * smooth_loss
                )

            # backward + optimizer step + scaler update (기존)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            # (선택) 로그에 smooth loss 출력
            print(f"[Epoch {epoch+1}/{num_epoch}] "
                f"ROI={roi_loss.item():.6f}  "
                f"Smooth={smooth_loss.item():.6f}  "
                f"Prior={prior_loss.item():.6f}  "
                f"Total={total_loss.item():.6f}")


            # --- slice-wise early stopping 업데이트 ---
            with torch.no_grad():
                improved = per_slice_loss < best_loss
                best_loss[improved] = per_slice_loss[improved]
                epochs_no_improve[~improved] += 1
                epochs_no_improve[improved] = 0
                active[epochs_no_improve > patience] = False

            # --- batch-wise projection trick ---
            with torch.no_grad():
                x.data = (1 - alpha_proj) * x.data + alpha_proj * x0
                c.data = (1 - alpha_proj) * c.data + alpha_proj * c0

            # print(f"[Epoch {epoch+1}/{num_epoch}] "
            #     f"Total={total_loss.item():.6f}  ROI={roi_loss.item():.6f}  "
            #     f"active_slices={active.sum().item()}")

            # 모든 slice 수렴 시 종료
            if active.sum() == 0:
                print("All slices converged → stopping.")
                break


                
            # # 8) 로그 출력
            # if local_rank == 0:
            #     print(f"[Epoch {epoch+1}/{num_epoch}] "
            #         f"Total={total_loss.item():.4f}  "
            #         f"Smooth={smooth_loss.item():.6f}(λ={lambda_smooth:.1f})  "
            #         f"Prior={prior_loss.item():.6f}(λ={lambda_prior:.2f})"
            #         f"Mse={roi_loss}")

        print("Optimization 완료")

            # 메타 추출
        # --- 메타 추출 (변경 없음) ---
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

        # --- 최적화된 batch 변수 사용 ---
        opt_xTs = x.detach()     # 전체 슬라이스를 batch로 묶은 optimized latent vectors[^opt_vars]
        opt_cs  = c.detach()     # 전체 슬라이스를 batch로 묶은 optimized conditions[^opt_vars]

        # --- 모델로 렌더링 후 정규화 ---
        with torch.no_grad():
            # optimized 이후
            out_after = base_model.render(opt_xTs, cond=opt_cs, T=t)   # named args 사용으로 명확히 지정[^render_args]
            pred_after = (out_after["sample"] + 1) / 2                # [-1,1] → [0,1]

            # 최적화 전 (before) 결과도 동일 T로 렌더링
            out_before = base_model.render(before_interp_xTs, cond=before_interp_cs, T=t)
            pred_before = (out_before["sample"] + 1) / 2

        # --- NumPy 배열로 변환 및 flip 처리 ---
        vol_after = pred_after.squeeze(1).cpu().numpy().astype(np.float32)
        vol_after = np.flip(vol_after, axis=1).copy()    # y축 뒤집기

        vol_pre = pred_before.squeeze(1).cpu().numpy().astype(np.float32)
        vol_pre = np.flip(vol_pre, axis=1).copy()

        # --- z 간격 보정 ---
        new_n = vol_after.shape[0]
        new_z = spacing3d[2] * (orig_n - 1) / (new_n - 1)

        # --- 원본 SAX volume도 동일 spacing/new_z 로 저장 ---
        itk_orig = sitk.GetImageFromArray(vol_pre)
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

        save_dir_a = f"/workspace/storage/kjh/dataset/cardiac/output/Inverse/subject_{sid}/frame_{frame_idx}/"
        os.makedirs(save_dir_a, exist_ok=True)
        save_path_a = os.path.join(
            save_dir_a,
            f"MR_Heart_{sid}_frame_{frame_idx}_final_{new_n}slice.nii.gz"
        )
        sitk.WriteImage(itk_after, save_path_a)
        print(f"[Rank {local_rank}] Saved optimized SAX volume → {save_path_a}")




if __name__=="__main__":
    # torchrun --nproc_per_node=4 LA_condition_all_pred_after.py
    main()