# slice 별 predict
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


# --- 하이퍼파라미터 및 Early Stopping 설정 ---
num_epochs    = 200
num_steps     = 1
base_lr       = 2e-3
lambda_roi    = 5.0
lambda_latent = 0.02
alpha_proj    = 0.001
lambda_smooth = 1.5

# --- Early Stopping 관련 상수 (각 slice별로 초기화할 것) ---
min_delta      = 1e-5   # loss가 이보다 적게 감소하면 '개선되지 않음'으로 간주
patience       = 5      # 5 epoch 연속 개선 없으면 중단

# --- Diffusion Model 파라미터 ---
t = 30           # noise level (fixed)
filling = 8      # interpolation할 slice 개수

# --- 처리할 Subject ID 리스트 ---
id_list = [
    148, 400, 769, 834, 243, 813, 600, 100, 708,
    342, 790, 823, 114, 760, 665, 812, 788, 515, 810,
    814, 786, 835, 182, 410, 724, 526, 704, 723, 387,
    776, 488, 711, 482, 190, 648, 692, 722, 664, 599
]

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
        "/storage/kjh/cardiac/DMCVR/generation/diffae_multi_PNU/"
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
    my_ids = id_list[local_rank::world_size]

    for sid in my_ids:
        # --- SAX (Short-Axis) 4D NIfTI 파일 경로 ---
        sax_path = (
            f"/storage/kjh/dataset/cardiac/PNU_cardiac/CINE/test/"
            f"middle_slice/sax/MR_Heart_{sid}_crop_sa.nii.gz"
        )
        # --- LAX (Long-Axis) 경로 템플릿 ---
        lax_dir_template      = (
            "/storage/icml_data_collection/Cardiac_database/PNUH/"
            "PNUH_cine_LAX_SAX_align_2/MR_Heart_{sid}/"
            "combine_1lax_cropped/combine_1lax_cropped_{frame}.nii.gz"
        )
        lax_mask_dir_template = (
            "/storage/icml_data_collection/Cardiac_database/PNUH/"
            "PNUH_cine_LAX_SAX_align_2/MR_Heart_{sid}/"
            "combine_1lax_mask_cropped_adjusted/"
            "combine_1lax_mask_cropped_adjusted_{frame}.nii.gz"
        )

        # --- SAX 4D 파일 로드 ---
        img_itk_sax = sitk.ReadImage(sax_path)
        vol_sax_all = sitk.GetArrayFromImage(img_itk_sax)  # [n_frames, n_slices, H, W]

        mx_sax_frame = np.percentile(vol_sax_all, 98)
        vol_frame_all = np.clip(vol_sax_all, 0, mx_sax_frame) / mx_sax_frame
        vol_frame_all = vol_frame_all * 2.0 - 1.0

        n_frames = [0, 13]
        for frame_idx in n_frames:
            # 1) 현재 frame의 SAX 볼륨 → numpy (z, H, W)
            vol_frame = vol_frame_all[frame_idx]  # [orig_n_slices, H, W]


            # 3) 좌우/상하 Flip (height 축 기준)
            vol_frame = np.flip(vol_frame, axis=-2).copy()

            # 4) 빈 슬라이스 제거
            nonzero_idx = [
                i for i in range(vol_frame.shape[0])
                if vol_frame[i].max() != 0
            ]
            vol_nz = vol_frame[nonzero_idx]  # [n_nz, H, W]
            n_nz = len(nonzero_idx)

            # --- 보간(interpolation) 준비 (CPU 환경) ---
            img_tensor_cpu  = torch.from_numpy(vol_nz).float().unsqueeze(1)  # CPU tensor
            cond_tensor_cpu = model.module.encode(img_tensor_cpu.to(device)).cpu()
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

            # --- (★추가) Optimize 전 “그냥 pred” 저장 ---
            # 전체 연속 latent/cond를 GPU로 한번에 올려 렌더링
            print("interp_xTs_cpu:",interp_xTs_cpu.shape)
            print("interp_conds_cpu:", interp_conds_cpu.shape)
            with torch.no_grad():
                interp_xTs_gpu   = interp_xTs_cpu.to(device)
                interp_conds_gpu = interp_conds_cpu.to(device)
                pred_before = model.module.render(
                    interp_xTs_gpu, cond=interp_conds_gpu, T=t
                )  # [new_n, 1, H, W]
                pred_before = (pred_before["sample"] + 1) / 2
                pred_before_np = pred_before.squeeze(1).cpu().numpy()  # [new_n, H, W]

            # 원래 intensity 범위로 복원
            pred_before_np = (pred_before_np + 1.0) / 2.0  # [0,1]
            pred_before_np = np.clip(pred_before_np, 0, 1) * mx_sax_frame  # [0, mx_sax_frame]

            # flip undo → 디스크에 올바른 좌표계로 저장
            pred_before_np = np.flip(pred_before_np, axis=1).copy()

            # SimpleITK 변환 및 affine 설정
            out_itk_before = sitk.GetImageFromArray(pred_before_np.astype(np.float32))
            orig_spacing_4d = img_itk_sax.GetSpacing()   # (sx, sy, sz, st)
            orig_origin_4d  = img_itk_sax.GetOrigin()    # (ox, oy, oz, ot)
            orig_direction_4d = img_itk_sax.GetDirection()  # 16 or 9 elements

            orig_z = orig_spacing_4d[2]
            orig_n = vol_frame.shape[0]
            new_z = orig_z * (orig_n - 1) / (new_n - 1)

            out_itk_before.SetSpacing((orig_spacing_4d[0], orig_spacing_4d[1], new_z))
            out_itk_before.SetOrigin((orig_origin_4d[0], orig_origin_4d[1], orig_origin_4d[2]))

            if len(orig_direction_4d) == 16:
                dir_3d = (
                    orig_direction_4d[0], orig_direction_4d[1], orig_direction_4d[2],
                    orig_direction_4d[4], orig_direction_4d[5], orig_direction_4d[6],
                    orig_direction_4d[8], orig_direction_4d[9], orig_direction_4d[10]
                )
            else:
                dir_3d = orig_direction_4d
            out_itk_before.SetDirection(dir_3d)

            save_dir_before = (
                f"/storage/kjh/dataset/cardiac/output/DiffAE/"
                f"subject_{sid}/frame_{frame_idx}/"
            )
            os.makedirs(save_dir_before, exist_ok=True)
            save_path_before = os.path.join(
                save_dir_before,
                f"MR_Heart_{sid}_frame_{frame_idx}_pred_before_opt_{new_n}slice.nii.gz"
            )
            sitk.WriteImage(out_itk_before, save_path_before)
            print(f"[Rank {local_rank}] Saved pre-optimization pred ({new_n} slices) → {save_path_before}")

            # GPU 메모리에서 렌더링 대량 텐서 삭제
            del interp_xTs_gpu, interp_conds_gpu, pred_before
            torch.cuda.empty_cache()

            # --- 3D 예측 볼륨 placeholder 초기화 ---
            vol_pred_full = np.zeros((new_n, vol_frame.shape[1], vol_frame.shape[2]), dtype=np.float32)

            # --- LAX & Mask 로드 (CPU에서만 보관) ---
            lax_path      = lax_dir_template.format(sid=sid, frame=frame_idx)
            lax_mask_path = lax_mask_dir_template.format(sid=sid, frame=frame_idx)

            gt_np_original = sitk.GetArrayFromImage(sitk.ReadImage(lax_path)).astype(np.float32)
            mask_original  = sitk.GetArrayFromImage(sitk.ReadImage(lax_mask_path)).astype(np.float32)

            mx_lax_frame = np.nanpercentile(gt_np_original, 99)
            gt_np = np.clip(gt_np_original, 0, mx_lax_frame) / mx_lax_frame
            gt_np = np.flip(gt_np, axis=-2).copy()       # flip
            mask = np.flip(mask_original, axis=-2).copy()  # flip

            # LAX GT/Mask 모두 CPU 텐서로 보관
            region_gt_cpu   = torch.tensor(gt_np, dtype=torch.float32)   # [z_lax, H, W] CPU
            region_mask_cpu = torch.tensor(mask,   dtype=torch.float32)   # [z_lax, H, W] CPU

            # --- 1) 메모리 리스트 초기화 ---
            # 각 슬라이스의 최신 latent & cond를 CPU 메모리에서 보관
            mem_x_cpu    = [interp_xTs_cpu[i:i+1].clone()   for i in range(new_n)]
            mem_cond_cpu = [interp_conds_cpu[i:i+1].clone() for i in range(new_n)]

            # --- 2) Slice별 최적화 시작 (new_n 개수만큼) ---
            for interp_idx in range(new_n):
                best_loss         = float('inf')
                epochs_no_improve = 0

                # 3) Slice별 latent & condition 초기값 설정 (GPU에 올려 최적화 변수)
                x_t_slice_gpu  = mem_x_cpu[interp_idx].to(device).detach().clone().requires_grad_(True)
                cond_slice_gpu = mem_cond_cpu[interp_idx].to(device).detach().clone().requires_grad_(True)
                x_t_slice_init  = x_t_slice_gpu.detach().clone()
                cond_slice_init = cond_slice_gpu.detach().clone()

                # 4) GT 및 Mask for this slice
                gt_gpu          = region_gt_cpu[interp_idx:interp_idx+1].to(device).unsqueeze(0)    # [1,1,H,W]
                region_mask_gpu = region_mask_cpu[interp_idx:interp_idx+1].to(device).unsqueeze(0) # [1,1,H,W]
                n_valid         = region_mask_gpu.sum().float().clamp(min=1.0)
                gt_roi          = gt_gpu * region_mask_gpu
                mu_gt           = gt_roi.sum() / n_valid
                sigma_gt        = torch.sqrt(((gt_roi - mu_gt) * region_mask_gpu).pow(2).sum() / n_valid + 1e-6)

                # 5) Optimizer & Scheduler (slice별로 한 번만 생성)
                optimizer = Adam([x_t_slice_gpu, cond_slice_gpu], lr=base_lr)
                scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

                # 6) (선택) Gradient Hook 등록
                def print_grad(name):
                    def hook(grad):
                        print(f"{name}.grad norm = {grad.norm().item():.6f}")
                    return hook

                x_t_slice_gpu.register_hook(print_grad("x_t_slice_gpu"))
                cond_slice_gpu.register_hook(print_grad("cond_slice_gpu"))

                scaler = torch.cuda.amp.GradScaler()

                # --- 7) 최적화 루프 (Epoch, Step) ---
                for epoch in range(num_epochs):
                    epoch_loss = 0.0

                    for step in range(num_steps):
                        with torch.cuda.amp.autocast():
                            # a) Forward pass
                            pred = model.module.render(x_t_slice_gpu, cond=cond_slice_gpu, T=t)  # [1,1,H,W]
                            pred = (pred["sample"] + 1) / 2

                            # b) ROI loss 계산 (Affine normalization 포함)
                            pred_roi   = pred * region_mask_gpu
                            mu_pred    = pred_roi.sum() / n_valid
                            sigma_pred = torch.sqrt(((pred_roi - mu_pred) * region_mask_gpu).pow(2).sum() / n_valid + 1e-6)
                            sigma_pred = torch.clamp(sigma_pred, min=1e-3)
                            pred_norm  = (pred - mu_pred) * (sigma_gt / sigma_pred) + mu_gt
                            pred_norm_roi = pred_norm * region_mask_gpu
                            roi_loss   = F.mse_loss(pred_norm_roi, gt_roi, reduction='sum') / n_valid

                            # c) Latent prior loss
                            latent_reg = ((x_t_slice_gpu - x_t_slice_init)**2).mean()
                            cond_reg   = ((cond_slice_gpu - cond_slice_init)**2).mean()
                            prior_loss = lambda_latent * (latent_reg + cond_reg)

                            # d) Smoothness term (이웃 슬라이스와의 MSE)
                            smooth_loss = 0.0
                            if interp_idx > 0:
                                prev = mem_x_cpu[interp_idx-1].to(device)
                                smooth_loss += F.mse_loss(x_t_slice_gpu, prev)
                            if interp_idx < new_n-1:
                                nxt = interp_xTs_cpu[interp_idx+1:interp_idx+2].to(device)
                                smooth_loss += F.mse_loss(x_t_slice_gpu, nxt)

                            # e) Total loss
                            loss = (lambda_roi * roi_loss
                                + prior_loss
                                + lambda_smooth * smooth_loss)
                            epoch_loss += loss.item()

                            # f) Backward & Optimizer step
                            optimizer.zero_grad()
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                        # g) 메모리 해제 및 캐시 정리
                        del pred
                        torch.cuda.empty_cache()

                        # h) Projection trick
                        with torch.no_grad():
                            x_t_slice_gpu.data  = (1 - alpha_proj) * x_t_slice_gpu.data  + alpha_proj * x_t_slice_init.data
                            cond_slice_gpu.data = (1 - alpha_proj) * cond_slice_gpu.data + alpha_proj * cond_slice_init.data

                        # i) Logging
                        print(
                            f"[Frame {frame_idx}][Slice {interp_idx}][E{epoch+1}S{step+1}] "
                            f"loss={loss.item():.6f} "
                            f"(roi={roi_loss:.6f}, prior={prior_loss.item():.6f}, smooth={smooth_loss.item():.6f})"
                        )

                    # --- 한 epoch이 끝난 뒤 평균 loss & Early Stopping 체크 ---
                    avg_epoch_loss = epoch_loss / num_steps
                    if best_loss - avg_epoch_loss > min_delta:
                        best_loss = avg_epoch_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        print(f"  → Improvement 없음: epochs_no_improve = {epochs_no_improve}/{patience}")

                    if epochs_no_improve >= patience:
                        print(
                            f"\nEarly stopping triggered! "
                            f"(Frame {frame_idx}, Slice {interp_idx}, Epoch {epoch+1})"
                        )
                        break

                    scheduler.step()

                # --- 8) 최적화 종료 후 메모리 업데이트 ---
                mem_x_cpu[interp_idx]    = x_t_slice_gpu.detach().cpu().clone()
                mem_cond_cpu[interp_idx] = cond_slice_gpu.detach().cpu().clone()

                # --- 최적화 완료 후 예측 저장 (각 보간된 slice별) ---
                with torch.no_grad():
                    final_pred = model.module.render(x_t_slice_gpu, cond=cond_slice_gpu, T=t)  # [1,1,H,W]
                    final_pred = (final_pred["sample"] + 1) / 2

                pred_np_2d = final_pred.squeeze().cpu().numpy()  # [H, W]
                vol_pred_full[interp_idx] = pred_np_2d

                print(f"[Frame {frame_idx}][Slice {interp_idx}] Assigned to vol_pred_full[{interp_idx}]")

                # ─── 메모리 사용량 체크 (1) ─────────────────
                gpu_idx = torch.cuda.current_device()
                alloc_before = torch.cuda.memory_allocated(device=gpu_idx)
                reserv_before = torch.cuda.memory_reserved(device=gpu_idx)
                print(
                    f"  → [GPU{gpu_idx}] After optimize (before delete): "
                    f"allocated = {alloc_before/1024**3:.3f} GiB, "
                    f"reserved = {reserv_before/1024**3:.3f} GiB"
                )

                # ——— GPU 메모리 해제 ————————————————
                del x_t_slice_gpu, cond_slice_gpu
                del gt_gpu, region_mask_gpu
                del final_pred, pred_np_2d
                del optimizer, scheduler
                torch.cuda.empty_cache()

                # ─── 메모리 사용량 체크 (2) ─────────────────
                alloc_after = torch.cuda.memory_allocated(device=gpu_idx)
                reserv_after = torch.cuda.memory_reserved(device=gpu_idx)
                print(
                    f"  → [GPU{gpu_idx}] After delete & empty_cache: "
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
                f"/storage/kjh/dataset/cardiac/output/Inverse/"
                f"subject_{sid}/frame_{frame_idx}/"
            )
            os.makedirs(save_dir_3d, exist_ok=True)
            save_path_3d = os.path.join(
                save_dir_3d,
                f"MR_Heart_{sid}_frame_{frame_idx}_optimized_interp_smooth_slice_{new_n}slice.nii.gz"
            )
            sitk.WriteImage(pred_itk_3d, save_path_3d)
            print(f"[Rank {local_rank}] Saved post-optimization volume ({new_n} slices) → {save_path_3d}\n")

            # --- 불필요한 CPU 변수 삭제 ---
            del interp_xTs_cpu, interp_conds_cpu
            del xTs_cpu, cond_tensor_cpu, img_tensor_cpu
            del region_gt_cpu, region_mask_cpu
            torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(world_size,),     # main(local_rank, world_size) 로 전달
        nprocs=world_size,
        join=True
    )
