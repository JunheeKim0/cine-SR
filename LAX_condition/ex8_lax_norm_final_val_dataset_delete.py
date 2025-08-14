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
import ants
import glob

# --- 하이퍼파라미터 및 Early Stopping 설정 ---
num_epochs    = 200
num_steps     = 1
base_lr       = 2e-3
# base_lr       = 4e-3
# lambda_roi    = 5.0
lambda_roi    = 5.0
lambda_latent = 0.02
lambda_smooth = 5.0
# lambda_smooth = 100.0
alpha_proj    = 0.001

# --- Early Stopping 관련 상수 (각 slice별로 초기화할 것) ---
min_delta      = 1e-5   # loss가 이보다 적게 감소하면 '개선되지 않음'으로 간주
patience       = 5      # 5 epoch 연속 개선 없으면 중단

# --- Diffusion Model 파라미터 ---
t = 30           # noise level (fixed)
filling = 8      # interpolation할 slice 개수

# val_dir = "/workspace/storage/kjh/dataset/cardiac/PNU_cardiac/CINE/40_exa/val"

# id_path_list = glob.glob(os.path.join(val_dir, '*.nii.gz'))
base_dir = "/workspace/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_normal_2"
id_path_list = os.listdir(base_dir)

n_frames = [0]

# --- Interpolation 함수 정의 ---
def lin_interpolate(slice_1, slice_2, num=filling):
    """Linear interpolation between two 조건 텐서"""
    alpha = 1.0 / (num - 1.0)
    out = []
    for i in range(num - 1):
        out.append(i * alpha * slice_2 + (1.0 - i * alpha) * slice_1)
    return out

def slerp_np(x0: np.ndarray, x1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical Linear intERPolation (numpy)"""
    theta = np.arccos(
        np.dot(x0.flatten(), x1.flatten()) /
        (np.linalg.norm(x0) * np.linalg.norm(x1) + 1e-8)
    )
    return (
        np.sin((1 - alpha) * theta) * x0 / (np.sin(theta) + 1e-8) +
        np.sin(alpha * theta) * x1 / (np.sin(theta) + 1e-8)
    )

def slerp_interpolate(slice_1, slice_2, num=filling):
    """Spherical interpolation between 두 latent numpy 배열"""
    alpha = 1.0 / (num - 1.0)
    out = []
    for i in range(num - 1):
        out.append(slerp_np(slice_1, slice_2, alpha * i))
    return out

def preprocess_n4(arr):
    img_ants = ants.from_numpy(arr)
    arr_n4   = ants.n4_bias_field_correction(img_ants).numpy()
    return arr_n4

def main(local_rank, world_size):
    # -----------------------
    # DistributedDataParallel 초기화
    # -----------------------
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12355',
        world_size=world_size,
        rank=local_rank
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # --- 모델 초기화 및 가중치 로드 ---
    conf = autoenc_base()
    conf.img_size = 128
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.model_name = ModelName.beatgans_autoenc
    conf.name = 'med256_autoenc'
    conf.make_model_conf()

    # 1) 원본 LitModel 초기화
    model = LitModel(conf)

    # 2) 체크포인트 불러오기 (CPU에 맵)
    checkpoint = torch.load(
        "/workspace/storage/kjh/cardiac/DMCVR/generation/diffae_multi_PNU/"
        "checkpoints_multi/med256_autoenc/1/epoch=438-step=756789.ckpt",
        map_location="cpu"
    )

    # 3) state_dict 로드
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # 4) EMA 모델 eval
    model.ema_model.eval()

    # 5) GPU 배정 후 DDP 래핑
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 각 GPU마다 처리할 subject ID 분할
    world_size = torch.cuda.device_count()
    # my_ids = id_path_list[local_rank::world_size]
    my_ids = id_path_list[::-1][local_rank::world_size]

    for my_id in my_ids:
        path = os.path.join(base_dir, my_id)

        for frame_idx in n_frames:
            # sid = os.path.basename(path).split('_')[2]
            # print(f"start! {sid} {frame_idx}")
            print(f"start! {my_id}")
            # --- SAX (Short-Axis) 4D NIfTI 파일 경로 ---
            sax_path = (
                f"/workspace/storage/kjh/dataset/cardiac/PNU_cardiac/CINE/normal/sax"
                f"/{my_id}_crop_sa.nii.gz"
            )
            sax_interp_path = (
                f"/workspace/storage/icml_data_collection/Cardiac_database/PNUH"
                f"/PNUH_cine_LAX_SAX_align_normal_2/{my_id}/sax_resample_linear_cropped/sax_resample_linear_cropped_{frame_idx}.nii.gz"
            )
            # --- LAX (Long-Axis) 경로 템플릿 ---
            lax_path      = (
                "/workspace/storage/icml_data_collection/Cardiac_database/PNUH"
                f"/PNUH_cine_LAX_SAX_align_normal_2/{my_id}/"
                f"combine_1lax_cropped/combine_1lax_cropped_{frame_idx}.nii.gz"
            )
            lax_mask_path = (
                "/workspace/storage/icml_data_collection/Cardiac_database/PNUH/"
                f"PNUH_cine_LAX_SAX_align_normal_2/{my_id}//combine_1lax_mask_cropped_adjusted/"
                f"combine_1lax_mask_cropped_adjusted_{frame_idx}.nii.gz"
            )


            # --- SAX 4D 파일 로드 ---
            img_itk_sax = sitk.ReadImage(sax_path)
            vol_sax_all = sitk.GetArrayFromImage(img_itk_sax)  # [n_frames, n_slices, H, W]
            print("vol_sax_all", vol_sax_all.shape)

            vol_sax_all = preprocess_n4(vol_sax_all)
            mx_sax_frame = np.percentile(vol_sax_all, 98)
            vol_frame_all = np.clip(vol_sax_all, 0, mx_sax_frame) / mx_sax_frame
            vol_frame_all = vol_frame_all * 2.0 - 1.0
        

            # 1) 현재 frame의 SAX 볼륨 → numpy (z, H, W)
            vol_frame = vol_frame_all[frame_idx]  # [orig_n_slices, H, W]

            # 3) 좌우/상하 Flip (height 축 기준)
            vol_frame = np.flip(vol_frame, axis=-2).copy()

            # 4) 빈 슬라이스 제거
            nonzero_idx = [
                i for i in range(vol_frame.shape[0])
                if vol_frame[i].max() != 0
            ]
            print("nonzero_idx", nonzero_idx)
            vol_nz = vol_frame[nonzero_idx]  # [n_nz, H, W]

            # --- 보간(interpolation) 준비 (CPU 환경) ---
            img_tensor_cpu  = torch.from_numpy(vol_nz).float().unsqueeze(1)  # CPU tensor
            print("img_tensor_cpu", img_tensor_cpu.shape)
            cond_tensor_cpu = model.module.encode(img_tensor_cpu.to(device)).cpu()
            print("cond_tensor_cpu", cond_tensor_cpu.shape)
            xTs_cpu = model.module.encode_stochastic(
                img_tensor_cpu.to(device),
                cond_tensor_cpu.to(device),
                T=t
            ).cpu()

            xTs_np  = xTs_cpu.numpy()    # [n_nz, latent_dim]
            cond_np = cond_tensor_cpu.numpy()  # [n_nz, cond_dim]

            # 7) 보간(interpolation) → CPU 메모리
            final_x = []
            final_c = []
            for i in range(xTs_np.shape[0] - 1):
                final_x.append(xTs_np[i])
                final_c.append(cond_np[i])
                mid_x = slerp_interpolate(xTs_np[i], xTs_np[i + 1], num=filling)
                mid_c = lin_interpolate(cond_np[i], cond_np[i + 1], num=filling)
                final_x.extend(mid_x)
                final_c.extend(mid_c)
            final_x.append(xTs_np[-1])
            final_c.append(cond_np[-1])

            interp_xTs_cpu   = torch.from_numpy(np.stack(final_x)).float()  # [new_n, latent_dim]
            interp_conds_cpu = torch.from_numpy(np.stack(final_c)).float()  # [new_n, cond_dim]
            new_n = interp_xTs_cpu.shape[0]  # (n_nz−1)*(filling−1) + n_nz

            print("new_n :", new_n)

            # --- 3D 예측 볼륨 placeholder 초기화 ---
            vol_pred_full = np.zeros((new_n, vol_frame.shape[1], vol_frame.shape[2]), dtype=np.float32)

            # --- LAX & Mask 로드 (CPU에서만 보관) ---
            # lax_path      = lax_dir_template.format(sid=sid, frame=frame_idx)
            # lax_mask_path = lax_mask_dir_template.format(sid=sid, frame=frame_idx)

            gt_np_original = sitk.GetArrayFromImage(sitk.ReadImage(lax_path)).astype(np.float32)
            mask_original  = sitk.GetArrayFromImage(sitk.ReadImage(lax_mask_path)).astype(np.float32)

            gt_np_original = preprocess_n4(gt_np_original)
            mx_lax_frame = np.nanpercentile(gt_np_original, 99)
            gt_np = np.clip(gt_np_original, 0, mx_lax_frame) / mx_lax_frame
            gt_np = np.flip(gt_np, axis=-2).copy()       # flip
            mask = np.flip(mask_original, axis=-2).copy()  # flip


       ################ ----- LAX normalization ----- #####################
            sax_interp_np_original = sitk.GetArrayFromImage(sitk.ReadImage(sax_interp_path)).astype(np.float32)

            # 2.1) clipping & normalization (LAX와 동일한 방식)
            mx_sax_frame = np.nanpercentile(sax_interp_np_original, 99)
            sax_interp_np = np.clip(sax_interp_np_original, 0, mx_sax_frame) / mx_sax_frame

            # 2.2) flip 및 복사
            sax_interp_np = np.flip(sax_interp_np, axis=-2).copy()

            # --- 3) ROI mask 정의 (boolean mask) ---
            roi = mask > 0  # ROI: Region of Interest[^1]

            # --- 4) ROI 내 통계치(mu, sigma) 계산 ---
            sax_roi_vals = sax_interp_np[roi]
            gt_roi_vals  = gt_np[roi]

            mu_sax, sigma_sax = sax_roi_vals.mean(), sax_roi_vals.std()   # SAX ROI 평균·표준편차
            mu_gt,  sigma_gt  = gt_roi_vals.mean(),  gt_roi_vals.std()    # GT ROI 평균·표준편차[^2]

            # --- 5) GT에 affine normalization 적용 ---
            gt_np_affine = gt_np.copy()
            gt_np_affine[roi] = (gt_np[roi] - mu_gt) / sigma_gt * sigma_sax + mu_sax



            # LAX GT/Mask 모두 CPU 텐서로 보관
            region_gt_cpu   = torch.tensor(gt_np_affine, dtype=torch.float32)   # [z_lax, H, W] CPU
            region_mask_cpu = torch.tensor(mask,   dtype=torch.float32)   # [z_lax, H, W] CPU


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
            optimizers = [
                torch.optim.Adam([x_slices[i], cond_slices[i]], lr=base_lr)
                for i in range(new_n)
            ]
            schedulers = [
                torch.optim.lr_scheduler.StepLR(optimizers[i], step_size=50, gamma=0.5)
                for i in range(new_n)
            ]

            # 8) GT 및 Mask batch 준비
            region_gt_batch   = region_gt_cpu.unsqueeze(1).to(device)       # [new_n,1,H,W]
            region_mask_batch = region_mask_cpu.unsqueeze(1).to(device)     # [new_n,1,H,W]
            n_valid_batch     = region_mask_batch.view(new_n, -1).sum(dim=1)  # [new_n]

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
            for interp_idx in range(new_n):  # 수정: 슬라이스 루프 추가
                # 수정: x_t_slice_gpu, cond_slice_gpu -> x_slices[interp_idx], cond_slices[interp_idx]
                with torch.no_grad():
                    final_pred = model.module.render(
                        x_slices[interp_idx], cond=cond_slices[interp_idx], T=t
                    )["sample"]  # [1,1,H,W]
                    final_pred = (final_pred + 1) / 2

                pred_np_2d = final_pred.squeeze().cpu().numpy()  # [H, W]
                vol_pred_full[interp_idx] = pred_np_2d

                print(f"[Frame {frame_idx}][Slice {interp_idx}] Assigned to vol_pred_full[{interp_idx}]")

                # ─── 메모리 사용량 체크 (1) ─────────────────
                gpu_idx = torch.cuda.current_device()
                alloc_before = torch.cuda.memory_allocated(device=gpu_idx)
                reserv_before = torch.cuda.memory_reserved(device=gpu_idx)
                print(
                    f"  → [GPU{gpu_idx}] After optimize (before cleanup): "
                    f"allocated = {alloc_before/1024**3:.3f} GiB, "
                    f"reserved = {reserv_before/1024**3:.3f} GiB"
                )

                # ——— GPU 메모리 해제 ————————————————
                # 수정: 개별 optimizer/scheduler, GT 변수는 코드 상에서 리스트와 배치 변수를 사용하므로 삭제 불필요
                del final_pred, pred_np_2d  # 수정: 예측 결과만 삭제
                torch.cuda.empty_cache()

                # ─── 메모리 사용량 체크 (2) ─────────────────
                alloc_after = torch.cuda.memory_allocated(device=gpu_idx)
                reserv_after = torch.cuda.memory_reserved(device=gpu_idx)
                print(
                    f"  → [GPU{gpu_idx}] After cleanup & empty_cache: "
                    f"allocated = {alloc_after/1024**3:.3f} GiB, "
                    f"reserved = {reserv_after/1024**3:.3f} GiB"
                )
                print(f"  → [GPU{gpu_idx}] Ready for next slice\n")

            # --- Slice 루프 종료 후: frame 전체 3D 예측 볼륨 NIfTI로 저장 ---
            # 1) 원본 Affine 정보 가져오기
            orig_spacing_4d = img_itk_sax.GetSpacing()
            orig_origin_4d  = img_itk_sax.GetOrigin()
            orig_direction_4d = img_itk_sax.GetDirection()

            # Flip undo 후 저장
            vol_pred_full = np.flip(vol_pred_full, axis=1).copy()

            # 2) SimpleITK 변환
            pred_itk_3d = sitk.GetImageFromArray(vol_pred_full.astype(np.float32))

            # 3) Z spacing 재계산
            orig_z = orig_spacing_4d[2]
            orig_n = vol_frame.shape[0]
            new_z = orig_z * (orig_n - 1) / (new_n - 1)

            # 4) Spacing/Origin/Direction 설정
            pred_itk_3d.SetSpacing((orig_spacing_4d[0], orig_spacing_4d[1], new_z))
            pred_itk_3d.SetOrigin((orig_origin_4d[0], orig_origin_4d[1], orig_origin_4d[2]))
            if len(orig_direction_4d) == 16:
                dir_3d = (
                    orig_direction_4d[0], orig_direction_4d[1], orig_direction_4d[2],
                    orig_direction_4d[4], orig_direction_4d[5], orig_direction_4d[6],
                    orig_direction_4d[8], orig_direction_4d[9], orig_direction_4d[10]
                )
            else:
                dir_3d = orig_direction_4d
            pred_itk_3d.SetDirection(dir_3d)

            save_dir_3d = (
                f"/workspace/storage/kjh/dataset/cardiac/output/Inverse/normal/"
                f"{my_id}/frame_{frame_idx}/"
            )
            os.makedirs(save_dir_3d, exist_ok=True)
            save_path_3d = os.path.join(
                save_dir_3d,
                f"{my_id}_frame_{frame_idx}_final_{new_n}slice.nii.gz"
            )
            sitk.WriteImage(pred_itk_3d, save_path_3d)
            print(f"[Rank {local_rank}] Saved post-optimization volume ({new_n} slices) → {save_path_3d}\n")

            # --- 불필요한 CPU 변수 삭제 ---
            del optimizers, schedulers, x_slices, cond_slices
            torch.cuda.empty_cache()
            # del interp_xTs_cpu, interp_conds_cpu
            # del xTs_cpu, cond_tensor_cpu, img_tensor_cpu
            # del region_gt_cpu, region_mask_cpu
            # torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(world_size,),     # main(local_rank, world_size) 로 전달
        nprocs=world_size,
        join=True
    )
