import os, sys
# 수정: 현재 파일의 상위 두 단계 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import SimpleITK as sitk
from templates import *
from tqdm import tqdm
import random

# --- 하이퍼파라미터 및 Early Stopping 설정 ---
num_epochs    = 200
num_steps     = 1
base_lr       = 1e-3
lambda_roi    = 6.0
lambda_latent = 0.02
lambda_smooth = 3.0
alpha_proj    = 0.001

min_delta      = 1e-5
patience       = 5

t = 30
filling = 15
    # 834, 843, 148, 400, 769, 243, 813, 600, 100, 708,
    # 342, 790, 823, 114, 760, 665, 812, 788, 515, 810,
    # 814, 786, 835, 182, 410, 724, 526, 704, 723, 387,
    # 776, 488, 711, 482, 190, 648, 692, 722, 664, 599
id_list = [
    114, 760, 665, 812, 788, 515, 810,
    814, 786, 835, 182 
]

def lin_interpolate(slice_1, slice_2, num_mid=filling):
    out = []
    for i in range(1, num_mid+1):
        alpha = i / (num_mid + 1.0)
        out.append((1.0 - alpha) * slice_1 + alpha * slice_2)
    return out

def slerp_np(x0: np.ndarray, x1: np.ndarray, alpha: float) -> np.ndarray:
    theta = np.arccos(
        np.dot(x0.flatten(), x1.flatten()) /
        (np.linalg.norm(x0) * np.linalg.norm(x1) + 1e-8)
    )
    return (
        np.sin((1 - alpha) * theta) * x0 / (np.sin(theta) + 1e-8) +
        np.sin(alpha * theta) * x1 / (np.sin(theta) + 1e-8)
    )

def slerp_interpolate(slice_1, slice_2, num_mid=filling):
    out = []
    for i in range(1, num_mid+1):
        alpha = i / (num_mid + 1.0)
        out.append(slerp_np(slice_1, slice_2, alpha))
    return out

def main(local_rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12355',
        world_size=world_size,
        rank=local_rank
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    conf = autoenc_base()
    conf.img_size = 128
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.model_name = ModelName.beatgans_autoenc
    conf.name = 'med256_autoenc'
    conf.make_model_conf()

    model = LitModel(conf)
    checkpoint = torch.load(
        "/workspace/storage/kjh/cardiac/DMCVR/generation/diffae_multi_PNU/"
        "checkpoints_multi/med256_autoenc/1/epoch=438-step=756789.ckpt",
        map_location="cpu"
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.ema_model.eval()
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    my_ids = id_list[local_rank::world_size]
    for sid in my_ids:
        sax_path = (
            f"/workspace/storage/kjh/dataset/cardiac/PNU_cardiac/CINE/test/"
            f"middle_slice/sax/MR_Heart_{sid}_crop_sa.nii.gz"
        )
        sax_interp_path = (
            "/workspace/storage/icml_data_collection/Cardiac_database/PNUH/"
            "PNUH_cine_LAX_SAX_align_2/MR_Heart_{sid}/"
            "sax_resample_linear_cropped/sax_resample_linear_cropped_{frame}.nii.gz"
        )
        lax_dir_template = (
            "/workspace/storage/icml_data_collection/Cardiac_database/PNUH/"
            "PNUH_cine_LAX_SAX_align_2/MR_Heart_{sid}/"
            "combine_1lax_cropped/combine_1lax_cropped_{frame}.nii.gz"
        )
        lax_mask_dir_template = (
            "/workspace/storage/icml_data_collection/Cardiac_database/PNUH/"
            "PNUH_cine_LAX_SAX_align_2/MR_Heart_{sid}/"
            "combine_1lax_mask_cropped_adjusted/"
            "combine_1lax_mask_cropped_adjusted_{frame}.nii.gz"
        )

        img_itk_sax = sitk.ReadImage(sax_path)
        vol_sax_all = sitk.GetArrayFromImage(img_itk_sax)
        mx_sax_frame = np.percentile(vol_sax_all, 98)
        vol_frame_all = np.clip(vol_sax_all, 0, mx_sax_frame) / mx_sax_frame
        vol_frame_all = vol_frame_all * 2.0 - 1.0
        n_frames = [14]

        for frame_idx in n_frames:
            vol_frame = np.flip(vol_frame_all[frame_idx], axis=-2).copy()
            nonzero_idx = [i for i in range(vol_frame.shape[0]) if vol_frame[i].max() != 0]
            vol_nz = vol_frame[nonzero_idx]

            N = vol_nz.shape[0]
            # 가운데 5개 slice 인덱스 범위 계산 (1 ≤ idx ≤ N-2 범위 내로 클램핑)
            mid = N // 2
            start = max(1, mid - 2)
            end   = min(N - 1, mid + 2)
            # 중간 5개 후보 인덱스 리스트
            candidates = list(range(start, end + 1))

            # remove_idx = random.randint(1, vol_nz.shape[0] - 2)
            # left_orig_z  = nonzero_idx[remove_idx - 1]
            # right_orig_z = nonzero_idx[remove_idx + 1]
            # vol_pair = np.stack([vol_nz[remove_idx - 1], vol_nz[remove_idx + 1]])

            # 랜덤으로 하나 선택
            remove_idx = random.choice(candidates)

            left_orig_z  = nonzero_idx[remove_idx - 1]
            right_orig_z = nonzero_idx[remove_idx + 1]
            vol_pair = np.stack([vol_nz[remove_idx - 1], vol_nz[remove_idx + 1]])

            # log 저장
            log_dir = "/workspace/storage/kjh/dataset/cardiac/output/middle_slice_delete/logs"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"removed_slices_rank{local_rank}.csv")

            # 최초 실행 시 헤더 추가
            if not os.path.exists(log_path):
                with open(log_path, "w") as f:
                    f.write("subject_id,frame_idx,removed_nonzero_idx\n")

             # CSV 포맷: subject_id,frame_idx,removed_nonzero_index
            with open(log_path, "a") as f:
                 f.write(f"{sid},{frame_idx},{remove_idx}\n")


            img_tensor_cpu  = torch.from_numpy(vol_pair).float().unsqueeze(1)
            cond_tensor_cpu = model.module.encode(img_tensor_cpu.to(device)).cpu()
            xTs_cpu = model.module.encode_stochastic(
                img_tensor_cpu.to(device),
                cond_tensor_cpu.to(device),
                T=t
            ).cpu()

            xTs_np  = xTs_cpu.numpy()
            cond_np = cond_tensor_cpu.numpy()

            final_x, final_c = [], []
            for i in range(xTs_np.shape[0] - 1):
                final_x.append(xTs_np[i]); final_c.append(cond_np[i])
                mid_x = slerp_interpolate(xTs_np[i], xTs_np[i + 1], num_mid=filling)
                mid_c = lin_interpolate(cond_np[i], cond_np[i + 1], num_mid=filling)
                final_x.extend(mid_x); final_c.extend(mid_c)
            final_x.append(xTs_np[-1]); final_c.append(cond_np[-1])

            interp_xTs_cpu   = torch.from_numpy(np.stack(final_x)).float()
            interp_conds_cpu = torch.from_numpy(np.stack(final_c)).float()
            new_n = interp_xTs_cpu.shape[0]

            vol_pred_full = np.zeros((new_n, vol_frame.shape[1], vol_frame.shape[2]), dtype=np.float32)

            # --- LAX & Mask 로드 ---
            lax_path      = lax_dir_template.format(sid=sid, frame=frame_idx)
            lax_mask_path = lax_mask_dir_template.format(sid=sid, frame=frame_idx)
            gt_np_original = sitk.GetArrayFromImage(sitk.ReadImage(lax_path)).astype(np.float32)
            mask_original  = sitk.GetArrayFromImage(sitk.ReadImage(lax_mask_path)).astype(np.float32)
            mx_lax_frame = np.nanpercentile(gt_np_original, 99)
            gt_np = np.clip(gt_np_original, 0, mx_lax_frame) / mx_lax_frame
            gt_np = np.flip(gt_np, axis=-2).copy()
            mask  = np.flip(mask_original, axis=-2).copy()

            factor = 8
            left_gt_z  = left_orig_z * factor
            right_gt_z = right_orig_z * factor
            gt_z_indices = np.round(
                np.linspace(left_gt_z, right_gt_z, new_n)
            ).astype(int)
            gt_np = gt_np[gt_z_indices]
            mask  = mask[gt_z_indices]

            sax_interp_np_original = sitk.GetArrayFromImage(
                sitk.ReadImage(sax_interp_path.format(sid=sid, frame=frame_idx))
            ).astype(np.float32)
            mx_sax_frame = np.nanpercentile(sax_interp_np_original, 99)
            sax_interp_np = np.clip(sax_interp_np_original, 0, mx_sax_frame) / mx_sax_frame
            sax_interp_np = np.flip(sax_interp_np, axis=-2).copy()[gt_z_indices]

            roi = mask > 0
            sax_roi_vals = sax_interp_np[roi]
            gt_roi_vals  = gt_np[roi]
            mu_sax, sigma_sax = sax_roi_vals.mean(), sax_roi_vals.std()
            mu_gt,  sigma_gt  = gt_roi_vals.mean(),  gt_roi_vals.std()

            gt_np_affine = gt_np.copy()
            gt_np_affine[roi] = (gt_np[roi] - mu_gt) / sigma_gt * sigma_sax + mu_sax

            region_gt_cpu   = torch.tensor(gt_np_affine, dtype=torch.float32)
            region_mask_cpu = torch.tensor(mask, dtype=torch.float32)

     ################### ======== Slice별 최적화 시작 (new_n 개수만큼) ========= ######################

            # --- Slice별 최적화 시작 (new_n 개수만큼) ---
            # 7) Slice별 파라미터 및 Optimizer/Scheduler 초기화
            x_slices = [
                interp_xTs_cpu[i:i+1].to(device)
                            .detach().clone().requires_grad_(True)
                for i in range(new_n)
            ]
            cond_slices = [
                interp_conds_cpu[i:i+1].to(device)
                                .detach().clone().requires_grad_(True)
                for i in range(new_n)
            ]
            x_slices_init = [x.detach().clone() for x in x_slices]
            cond_slices_init = [c.detach().clone() for c in cond_slices]

            mid_indices = list(range(1, new_n-1))


            optimizers = [
                torch.optim.Adam([x_slices[i], cond_slices[i]], lr=base_lr)
                for i in mid_indices
            ]
            schedulers = [
                torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
                for opt in optimizers
            ]

            # 8) GT 및 Mask batch 준비
            region_gt_batch   = region_gt_cpu.unsqueeze(1).to(device)       # [new_n,1,H,W]
            region_mask_batch = region_mask_cpu.unsqueeze(1).to(device)     # [new_n,1,H,W]
            n_valid_batch     = region_mask_batch.view(new_n, -1).sum(dim=1)  # [new_n]
            
            # mid-slices에 대응하는 GT/Mask만 뽑아서
            region_gt_mid     = region_gt_batch[mid_indices]                # [M,1,H,W]
            region_mask_mid   = region_mask_batch[mid_indices]
            n_valid_mid       = region_mask_mid.view(len(mid_indices), -1).sum(dim=1)

            # --- Vectorized batch 최적화 루프 ---
            # 7\.1) AMP 스케일러 초기화  # 수정: scaler 재추가
            scaler = torch.cuda.amp.GradScaler()

            # Early Stopping 변수 초기화  # 수정: epoch마다 best_loss 초기화는 유지
            best_loss = float('inf')
            epochs_no_improve = 0
            
            for epoch in range(num_epochs):

                # 1) 모든 slice optimizer의 gradient 초기화  # 수정: step 루프 제거, epoch 단위
                for opt in optimizers:
                    opt.zero_grad()

                # 2) forward pass: batch로 묶어 처리
                x_batch = torch.cat(x_slices, dim=0)          # [new_n,C,H,W]
                cond_batch = torch.cat(cond_slices, dim=0)    # [new_n,cond_dim]
                with torch.cuda.amp.autocast():
                    pred_batch = model.module.render(
                        x_batch, cond=cond_batch, T=t
                    )["sample"]
                    pred_batch = (pred_batch + 1) / 2         # [new_n,1,H,W]

                    # ROI loss vectorized (unchanged)

                    roi_losses   = F.mse_loss(
                        pred_batch * region_mask_batch,
                        region_gt_batch,
                        reduction="none"
                    ).view(new_n, -1).sum(dim=1) / n_valid_batch

                    # Prior loss vectorized (unchanged)
                    x_init_batch    = torch.cat(x_slices_init, dim=0)
                    cond_init_batch = torch.cat(cond_slices_init, dim=0)
                    latent_reg      = ((x_batch - x_init_batch)
                                        .view(new_n, -1)**2).mean(dim=1)
                    cond_reg        = ((cond_batch - cond_init_batch)
                                        .view(new_n, -1)**2).mean(dim=1)
                    prior_losses    = lambda_latent * (latent_reg + cond_reg)

                    # Smooth loss vectorized (unchanged)
                    pred_pad         = torch.cat([pred_batch[:1], pred_batch, pred_batch[-1:]], dim=0)  # [new_n+2,1,H,W]
                    smooth_left_img  = ((pred_batch - pred_pad[:-2])
                                        .view(new_n, -1)**2).mean(dim=1)
                    smooth_right_img = ((pred_batch - pred_pad[2:])
                                        .view(new_n, -1)**2).mean(dim=1)
                    smooth_losses    = smooth_left_img + smooth_right_img

                    # Total loss per slice and sum
                    loss_per_slice  = (
                        lambda_roi * roi_losses + prior_losses + lambda_smooth * smooth_losses
                    )
                    total_loss      = loss_per_slice.sum()        # 수정

                # 3) backward 한 번만 호출  # 수정
                scaler.scale(total_loss).backward()

                # 4) 각 slice optimizer step  # 수정
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()

                # 5) 각 slice scheduler step  # 수정
                for sch in schedulers:
                    sch.step()

                # 6) Projection 트릭 (batch)  # 수정
                with torch.no_grad():
                    for i in range(new_n):
                        x_slices[i].data = (
                            (1 - alpha_proj) * x_slices[i].data + alpha_proj * x_slices_init[i].data
                        )
                        cond_slices[i].data = (
                            (1 - alpha_proj) * cond_slices[i].data + alpha_proj * cond_slices_init[i].data
                        )

                # --- Early Stopping 체크 및 로깅 ---  # unchanged
                avg_epoch_loss = total_loss.item() / new_n
                if best_loss - avg_epoch_loss > min_delta:
                    best_loss = avg_epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at Epoch {epoch+1}")
                    break

                print(
                    f"[Frame {frame_idx}][Epoch {epoch+1}] roi_loss = {roi_losses.sum()} prior_loss = {prior_losses.sum()} smooth_losse = {smooth_losses.sum()} total_loss={total_loss.item():.6f}")

        ################### --- 최적화 완료 후 예측 저장 (각 보간된 slice별) --- #######################
            for interp_idx in range(new_n):
                with torch.no_grad():
                    final_pred = model.module.render(
                        x_slices[interp_idx], cond=cond_slices[interp_idx], T=t
                    )["sample"]
                vol_pred_full[interp_idx] = ((final_pred + 1) / 2).squeeze().cpu().numpy()

                print(f"[Frame {frame_idx}][Slice {interp_idx}] Assigned to vol_pred_full[{interp_idx}]")

            # ——— GPU 메모리 해제 ————————————————
            # 수정: 개별 optimizer/scheduler, GT 변수는 코드 상에서 리스트와 배치 변수를 사용하므로 삭제 불필요
            del final_pred
            torch.cuda.empty_cache()

            # --- Slice 루프 종료 후: frame 전체 3D 예측 볼륨 NIfTI로 저장 ---
            # 1) 원본 Affine 정보 (frame 루프 밖에서 한 번만 로드 권장)
            orig_spacing_4d   = img_itk_sax.GetSpacing()   # (sx, sy, sz)
            orig_origin_4d    = img_itk_sax.GetOrigin()    # (ox, oy, oz)
            orig_direction_4d = img_itk_sax.GetDirection() # length 9 or 16

            # Direction matrix 재사용
            if len(orig_direction_4d) == 16:
                dir_3d = (
                    orig_direction_4d[0],orig_direction_4d[1],orig_direction_4d[2],
                    orig_direction_4d[4],orig_direction_4d[5],orig_direction_4d[6],
                    orig_direction_4d[8],orig_direction_4d[9],orig_direction_4d[10]
                )
            else:
                dir_3d = orig_direction_4d

            # 삭제된 두 슬라이스 간 실제 거리(distance) 및 new_z/origin_z 계산
            distance      = (right_orig_z - left_orig_z) * orig_spacing_4d[2]  # mm 단위 총 거리[^1]
            new_z         = distance / (new_n - 1)                            # mm 단위 new spacing[^2]
            origin_z_base = orig_origin_4d[2] + left_orig_z * orig_spacing_4d[2]  # 시작 origin z

            # --- ★ Pre-optimization 예측 및 저장 ★
            # (1) vol_preopt 정의 및 예측
            vol_preopt = np.zeros((new_n, vol_frame.shape[1], vol_frame.shape[2]), dtype=np.float32)  # [new_n, H, W][^3]
            for i in range(new_n):
                with torch.no_grad():
                    sample_out = model.module.render(
                        interp_xTs_cpu[i:i+1].to(device),
                        cond=interp_conds_cpu[i:i+1].to(device),
                        T=t
                    )["sample"]  # [1,1,H,W]
                vol_preopt[i] = ((sample_out + 1) / 2).squeeze().cpu().numpy()

            # (2) flip undo
            vol_preopt = np.flip(vol_preopt, axis=1).copy()

            # (3) SimpleITK 변환 & affine 설정
            pred_preopt_itk = sitk.GetImageFromArray(vol_preopt.astype(np.float32))
            pred_preopt_itk.SetSpacing((orig_spacing_4d[0], orig_spacing_4d[1], new_z))
            pred_preopt_itk.SetOrigin((orig_origin_4d[0], orig_origin_4d[1], origin_z_base))
            pred_preopt_itk.SetDirection(dir_3d)

            # (4) 저장
            save_dir_pre = f"/workspace/storage/kjh/dataset/cardiac/output/middle_slice_delete/DiffAE"
            os.makedirs(save_dir_pre, exist_ok=True)
            save_path_preopt = os.path.join(
                save_dir_pre,
                f"MR_Heart_{sid}_frame_{frame_idx}_preopt_{new_n}slice.nii.gz"
            )
            sitk.WriteImage(pred_preopt_itk, save_path_preopt)
            print(f"[Rank {local_rank}] Saved pre-opt volume → {save_path_preopt}")

            # --- ★ Post-optimization 예측 및 저장 ★
            # (1) flip undo
            vol_post = np.flip(vol_pred_full, axis=1).copy()

            # (2) SimpleITK 변환 & same affine
            pred_itk_3d = sitk.GetImageFromArray(vol_post.astype(np.float32))
            pred_itk_3d.SetSpacing((orig_spacing_4d[0], orig_spacing_4d[1], new_z))
            pred_itk_3d.SetOrigin((orig_origin_4d[0], orig_origin_4d[1], origin_z_base))
            pred_itk_3d.SetDirection(dir_3d)
            
            # (3) 저장
            save_dir_post = f"/workspace/storage/kjh/dataset/cardiac/output/middle_slice_delete/Inverse_new"
            os.makedirs(save_dir_post, exist_ok=True)
            save_path_post = os.path.join(
                save_dir_post,
                f"MR_Heart_{sid}_frame_{frame_idx}_final_{new_n}slice.nii.gz"
            )
            sitk.WriteImage(pred_itk_3d, save_path_post)
            print(f"[Rank {local_rank}] Saved post-opt volume ({new_n} slices) → {save_path_post}")
            # 불필요한 변수 정리
            del optimizers, schedulers, x_slices, cond_slices
            torch.cuda.empty_cache()


    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
