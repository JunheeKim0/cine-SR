from templates import *
import matplotlib.pyplot as plt
import numpy as np
from math import acos, sin
import SimpleITK as sitk
from tqdm import tqdm
import sys

import torch
import torch.nn.functional as F
import kornia.filters as KF
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from skimage.metrics import structural_similarity as ssim_fn, peak_signal_noise_ratio as psnr_fn

# --- 하이퍼파라미터 및 Early Stopping 설정 ---
num_epochs    = 50
num_steps     = 1
base_lr       = 2e-3
lambda_roi    = 5.0
lambda_latent = 0.02
alpha_proj    = 0.001

# --- Early Stopping 관련 상수 (각 slice별로 초기화할 것) ---
min_delta      = 1e-4      # loss가 이보다 적게 감소하면 '개선되지 않음'으로 간주
patience       = 5         # 5 epoch 연속 개선 없으면 중단

# --- GPU 설정 및 device 정의 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Diffusion Model 파라미터 ---
t = 30              # noise level (fixed) 
filling = 8         # interpolation할 slice 개수

# --- 처리할 Subject ID 리스트 ---
id_list = [
    148, 400, 769, 834, 243, 813, 600, 100, 708,
    342, 790, 823, 114, 760, 665, 812, 788, 515, 810,
    814, 786, 835, 182, 410, 724, 526, 704, 723, 387,
    776, 488, 711, 482, 190, 648, 692, 722, 664, 599
]

# --- Subject별 반복 시작 ---
for sid in id_list: 

    # --- SAX (Short-Axis) 4D NIfTI 파일 경로 ---
    sax_path = f"/storage/kjh/dataset/cardiac/PNU_cardiac/CINE/test/middle_slice/sax/MR_Heart_{sid}_crop_sa.nii.gz"
    # --- LAX (Long-Axis) 경로 템플릿 ---
    lax_dir_template       = "/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{sid}/combine_1lax_cropped/combine_1lax_cropped_{frame}.nii.gz"
    lax_mask_dir_template  = "/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{sid}/combine_1lax_mask_cropped_adjusted/combine_1lax_mask_cropped_adjusted_{frame}.nii.gz"

    # --- 모델 초기화 및 가중치 로드 ---
    conf = autoenc_base()  
    conf.img_size = 128
    conf.net_ch = 128
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.model_name = ModelName.beatgans_autoenc
    conf.name = 'med256_autoenc'
    conf.make_model_conf()

    model = LitModel(conf)
    state = torch.load(
        f"/storage/kjh/cardiac/DMCVR/generation/diffae_multi_PNU/checkpoints_multi/med256_autoenc/1/epoch=438-step=756789.ckpt", 
        map_location=device
    )
    model.load_state_dict(state['state_dict'], strict=True)
    model.ema_model.eval()
    model.ema_model.to(device)

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
            np.dot(x0.flatten(), x1.flatten()) 
            / (np.linalg.norm(x0) * np.linalg.norm(x1) + 1e-8)
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

    # --- SAX 4D 파일 로드 및 전체 Normalization/Flip --- 
    img_itk_sax = sitk.ReadImage(sax_path)
    vol_sax_all = sitk.GetArrayFromImage(img_itk_sax)  # shape: [n_frames, n_slices, H, W]

    # --- Frame별 반복 시작 --- 
    n_frames = [0, 13]
    for frame_idx in n_frames:
        # 1) 현재 frame의 SAX 볼륨 → numpy (z, H, W)
        vol_frame = vol_sax_all[frame_idx]  # shape: [orig_n_slices, H, W]

        # 2) Intensity normalize (percentile 98) 및 [-1,1] 스케일
        mx_sax_frame = np.percentile(vol_frame, 98)
        vol_frame = np.clip(vol_frame, 0, mx_sax_frame) / mx_sax_frame
        vol_frame = vol_frame * 2.0 - 1.0
        # 3) 좌우/상하 Flip (height 축 기준)
        vol_frame = np.flip(vol_frame, axis=-2).copy()

        # 4) 빈 슬라이스(모두 0) 제거
        nonzero_idx = [i for i in range(vol_frame.shape[0]) if vol_frame[i].max() != 0]
        vol_nz = vol_frame[nonzero_idx]  # shape: [n_nz, H, W]
        n_nz = len(nonzero_idx)          # 실제 non-zero slice 개수

        # --- 보간(interpolation) 준비 ---
        img_tensor  = torch.from_numpy(vol_nz).float().unsqueeze(1).to(device)  
        cond_tensor = model.encode(img_tensor)                                 
        xTs         = model.encode_stochastic(img_tensor, cond_tensor, T=t)   

        xTs_np  = xTs.cpu().numpy()    # shape: [n_nz, latent_dim]
        cond_np = cond_tensor.cpu().numpy()  # shape: [n_nz, cond_dim]

        # 7) 보간(interpolation)
        final_x = []
        final_c = []
        for i in range(xTs_np.shape[0] - 1):
            final_x.append(xTs_np[i]); final_c.append(cond_np[i])
            mid_x = slerp_interpolate(xTs_np[i], xTs_np[i + 1], num=filling)
            mid_c = lin_interpolate(cond_np[i],   cond_np[i + 1],   num=filling)
            final_x.extend(mid_x); final_c.extend(mid_c)
        final_x.append(xTs_np[-1]); final_c.append(cond_np[-1])

        interp_xTs   = torch.from_numpy(np.stack(final_x)).float().to(device)
        interp_conds = torch.from_numpy(np.stack(final_c)).float().to(device)
        # interp_xTs.shape = [total_interp, latent_dim]
        # interp_conds.shape = [total_interp, cond_dim]

        # --- (★추가) Optimize 전 “그냥 pred” 이미지를 저장 --- 
        # 보간된 전체 시퀀스로 바로 렌더링
        with torch.no_grad():
            pred_before_np = model.render(interp_xTs, cond=interp_conds, T=t).squeeze(1).cpu().numpy()  # shape: [new_n, H, W]

        pred_before_np = pred_before_np * mx_sax_frame  # [0, mx_sax_frame]
        pred_before_np = np.flip(pred_before_np, axis=1).copy()

        # SimpleITK 이미지로 변환
        out_itk_before = sitk.GetImageFromArray(pred_before_np.astype(np.float32))  # shape: [new_n, H, W]

        # 1) 원본 Affine 정보(X/Y spacing, origin, direction) 가져오기
        orig_spacing_4d = img_itk_sax.GetSpacing()   # (sx, sy, sz, st)
        orig_origin_4d  = img_itk_sax.GetOrigin()    # (ox, oy, oz, ot)
        orig_direction_4d = img_itk_sax.GetDirection()  # 16개 요소거나 9개 요소

        # 2) Z spacing 재계산 (전체 물리적 길이 보존)
        orig_z = orig_spacing_4d[2]
        orig_n = vol_frame.shape[0]          # 원본 slice 개수
        new_n = (n_nz - 1) * (filling - 1) + n_nz     # 보간 후 slice 개수
        new_z = orig_z * (orig_n-1) / (new_n-1)

        # 3) Spacing 설정 (X/Y는 원본 그대로, Z는 new_z)
        out_itk_before.SetSpacing((orig_spacing_4d[0], orig_spacing_4d[1], new_z))

        # 4) Origin 설정 (Z origin도 그대로 사용)
        out_itk_before.SetOrigin((orig_origin_4d[0], orig_origin_4d[1], orig_origin_4d[2]))

        # 5) Direction 설정 (4D → 3D 변환: 앞 9개 요소만 추출)
        if len(orig_direction_4d) == 16:
            dir_3d = (
                orig_direction_4d[0], orig_direction_4d[1], orig_direction_4d[2],
                orig_direction_4d[4], orig_direction_4d[5], orig_direction_4d[6],
                orig_direction_4d[8], orig_direction_4d[9], orig_direction_4d[10]
            )
        else:
            dir_3d = orig_direction_4d
        out_itk_before.SetDirection(dir_3d)

        # 6) 저장 경로 및 파일명 생성 (Optimize 전 pred)
        save_dir_before = f"/storage/kjh/dataset/cardiac/output/DiffAE/subject_{sid}/frame_{frame_idx}/"
        os.makedirs(save_dir_before, exist_ok=True)
        save_path_before = os.path.join(
            save_dir_before,
            f"MR_Heart_{sid}_frame_{frame_idx}_pred_before_opt_{new_n}slice.nii.gz"
        )
        sitk.WriteImage(out_itk_before, save_path_before)
        print(f"Saved pre-optimization pred volume (총 {new_n} slice) → {save_path_before}")

        # --- (기존) 이 frame의 3D 예측 볼륨 placeholder 초기화 --- 
        vol_pred_full = np.zeros((new_n, vol_frame.shape[1], vol_frame.shape[2]), dtype=np.float32)

        # --- 해당 frame의 LAX & Mask 로드 ---
        lax_path      = lax_dir_template.format(sid=sid, frame=frame_idx)
        lax_mask_path = lax_mask_dir_template.format(sid=sid, frame=frame_idx)

        gt_np_original = sitk.GetArrayFromImage(
            sitk.ReadImage(lax_path)
        ).astype(np.float32)  # shape: [z, H, W]

        mask_original  = sitk.GetArrayFromImage(
            sitk.ReadImage(lax_mask_path)
        ).astype(np.float32)  # shape: [z, H, W]

        # 5) LAX intensity normalize (exclude NaN), [0,1] 스케일
        mx_lax_frame = np.nanpercentile(gt_np_original, 99)
        gt_np = np.clip(gt_np_original, 0, mx_lax_frame) / mx_lax_frame

        gt_np = np.flip(gt_np, axis=-2).copy()       # flip
        mask = np.flip(mask_original, axis=-2).copy()  # flip

        # --- LAX 전체 볼륨 텐서 & Mask 전체 텐서 ---
        region_gt_all      = torch.tensor(gt_np, dtype=torch.float32).to(device)   # shape: [z, H, W]
        region_mask_all    = torch.tensor(mask_original, dtype=torch.float32).to(device)  # shape: [z, H, W]

        # --- Slice별 최적화 시작 (전체 보간된 slice 개수로 확장) ---
        for interp_idx in range(new_n):
            # --- 초기화 (각 slice마다 초기화 필수) ---
            best_loss         = float('inf')
            epochs_no_improve = 0
            loss_history      = []

            # 8) Slice별 latent & condition 초기값 설정 (requires_grad 설정)
            x_t_slice       = interp_xTs[interp_idx:interp_idx+1].detach().clone().requires_grad_(True).to(device)
            cond_slice      = interp_conds[interp_idx:interp_idx+1].detach().clone().requires_grad_(True).to(device)
            x_t_slice_init  = x_t_slice.detach().clone()
            cond_slice_init = cond_slice.detach().clone()

            # 9) GT 및 Mask for this slice (원본 non-zero slice에만 ROI loss 적용; 중간 보간 slice는 ROI loss 없이 prior만)
            if interp_idx % filling == 0:
                orig_slice_idx = nonzero_idx[interp_idx // filling]
                gt       = region_gt_all[orig_slice_idx:orig_slice_idx+1].unsqueeze(0).to(device)       # [1,1,H,W]
                region_mask = region_mask_all[orig_slice_idx:orig_slice_idx+1].unsqueeze(0)            # [1,1,H,W]
                n_valid = region_mask.sum().float()
                gt_roi  = gt * region_mask
                mu_gt   = gt_roi.sum() / n_valid
                sigma_gt = torch.sqrt(((gt_roi - mu_gt) * region_mask).pow(2).sum() / n_valid + 1e-6)
            else:
                gt = None
                region_mask = None
                n_valid = None
                mu_gt = None
                sigma_gt = None

            # 10) Optimizer & Scheduler 설정
            optimizer = Adam([x_t_slice, cond_slice], lr=base_lr)
            scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

            # 11) Gradient Hook 등록 (optional: gradient norm 출력용)
            def print_grad(name):
                def hook(grad):
                    print(f"{name}.grad norm = {grad.norm().item():.6f}")
                return hook

            x_t_slice.register_hook(print_grad("x_t_slice"))
            cond_slice.register_hook(print_grad("cond_slice"))

            # --- 최적화 루프 (Epoch, Step) ---
            for epoch in range(num_epochs):
                epoch_loss = 0.0

                for step in range(num_steps):
                    # 1) Forward pass: 예측 생성
                    pred = model.render(x_t_slice, cond=cond_slice, T=t)  # [1, H, W] or [1,1,H,W]

                    # 2) ROI 통계 및 Affine normalization
                    if region_mask is not None:
                        pred_roi = pred * region_mask
                        mu_pred = pred_roi.sum() / n_valid
                        sigma_pred = torch.sqrt(((pred_roi - mu_pred) * region_mask).pow(2).sum() / n_valid + 1e-6)
                        sigma_pred = torch.clamp(sigma_pred, min=1e-3)
                        pred_norm     = (pred - mu_pred) * (mu_gt / sigma_pred) + mu_gt
                        pred_norm_roi = pred_norm * region_mask
                        roi_loss = F.mse_loss(pred_norm_roi, gt_roi, reduction='sum') / n_valid
                    else:
                        roi_loss = 0.0

                    # 3) Latent prior loss
                    latent_reg = ((x_t_slice - x_t_slice_init) ** 2).mean()
                    cond_reg   = ((cond_slice - cond_slice_init) ** 2).mean()
                    prior_loss = lambda_latent * (latent_reg + cond_reg)

                    # 4) Total loss
                    loss = lambda_roi * roi_loss + prior_loss
                    epoch_loss += loss.item() if isinstance(loss, torch.Tensor) else loss

                    # 5) Backward & Optimizer step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 6) Projection: 초기값 쪽으로 살짝 밀어주는 트릭
                    with torch.no_grad():
                        x_t_slice.data  = (1 - alpha_proj) * x_t_slice.data  + alpha_proj * x_t_slice_init
                        cond_slice.data = (1 - alpha_proj) * cond_slice.data + alpha_proj * cond_slice_init

                    # 7) Logging (원본 non-zero slice 위치일 때만 로그에 slice 번호 표시)
                    if interp_idx % filling == 0:
                        print(f"[Frame {frame_idx}][Orig Slice {orig_slice_idx} (Interp {interp_idx})][E{epoch+1}S{step+1}] "
                              f"loss={loss.item():.6f} (roi={roi_loss:.6f}, prior={prior_loss.item():.6f})")
                    else:
                        print(f"[Frame {frame_idx}][Interp Slice {interp_idx}][E{epoch+1}S{step+1}] "
                              f"loss={loss.item():.6f} (roi=0.000000, prior={prior_loss.item():.6f})")

                # --- 한 epoch이 끝난 뒤 평균 loss 계산 & Early Stopping 체크 ---
                avg_epoch_loss = epoch_loss / num_steps
                if best_loss - avg_epoch_loss > min_delta:
                    best_loss = avg_epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"  → Improvement 없음: epochs_no_improve = {epochs_no_improve}/{patience}")

                if epochs_no_improve >= patience:
                    if interp_idx % filling == 0:
                        print(f"\nEarly stopping triggered! (Frame {frame_idx}, Orig Slice {orig_slice_idx}, Epoch {epoch+1})")
                    else:
                        print(f"\nEarly stopping triggered! (Frame {frame_idx}, Interp Slice {interp_idx}, Epoch {epoch+1})")
                    break

                scheduler.step()

            # --- 최적화 완료 후 예측 저장 (각 보간된 slice별) → vol_pred_full에 할당 ---
            with torch.no_grad():
                final_pred = model.render(x_t_slice, cond=cond_slice, T=t)

            pred_np_2d = final_pred.squeeze().detach().cpu().numpy()  # shape: [H, W]


            vol_pred_full[interp_idx] = pred_np_2d  # 보간 인덱스 위치에 할당
            if interp_idx % filling == 0:
                print(f"[Frame {frame_idx}][Orig Slice {orig_slice_idx}] 예측 결과를 vol_pred_full[{interp_idx}]에 할당")
            else:
                print(f"[Frame {frame_idx}][Interp Slice {interp_idx}] 예측 결과를 vol_pred_full[{interp_idx}]에 할당")

        # --- Slice 루프 종료 후: frame 전체 3D 예측 볼륨 NIfTI로 저장 ---
        # 1) 원본 Affine 정보(X/Y spacing, origin, direction) 가져오기
        orig_spacing_4d = img_itk_sax.GetSpacing()   # (sx, sy, sz, st)
        orig_origin_4d  = img_itk_sax.GetOrigin()    # (ox, oy, oz, ot)
        orig_direction_4d = img_itk_sax.GetDirection()  # 16개 요소거나 9개 요소

        vol_pred_full = np.flip(vol_pred_full, axis=1).copy()

        # 2) SimpleITK 이미지로 변환 (bo간된 총 slice 개수)
        pred_itk_3d = sitk.GetImageFromArray(vol_pred_full.astype(np.float32))  # shape: [new_n, H, W]

        # 3) Z spacing 재계산 (전체 물리적 길이 보존)
        orig_z = orig_spacing_4d[2]
        orig_n = vol_frame.shape[0]          # 원본 slice 개수
        new_n = (n_nz - 1) * (filling - 1) + n_nz     # 보간 후 slice 개수
        new_z = orig_z * (orig_n-1) / (new_n-1)

        # 4) Spacing 설정 (X/Y는 원본 그대로, Z는 new_z)
        pred_itk_3d.SetSpacing((orig_spacing_4d[0], orig_spacing_4d[1], new_z))

        # 5) Origin 설정 (X/Y/Z origin 중 Z origin 그대로 사용)
        pred_itk_3d.SetOrigin((orig_origin_4d[0], orig_origin_4d[1], orig_origin_4d[2]))

        # 6) Direction 설정 (4D → 3D 변환: 앞 9개 요소만 추출)
        if len(orig_direction_4d) == 16:
            dir_3d = (
                orig_direction_4d[0], orig_direction_4d[1], orig_direction_4d[2],
                orig_direction_4d[4], orig_direction_4d[5], orig_direction_4d[6],
                orig_direction_4d[8], orig_direction_4d[9], orig_direction_4d[10]
            )
        else:
            dir_3d = orig_direction_4d
        pred_itk_3d.SetDirection(dir_3d)

        # 7) 저장 경로 및 파일명 생성 (Optimize 후)
        save_dir_3d = f"/storage/kjh/dataset/cardiac/output/Inverse/subject_{sid}/frame_{frame_idx}/"
        os.makedirs(save_dir_3d, exist_ok=True)
        save_path_3d = os.path.join(
            save_dir_3d,
            f"MR_Heart_{sid}_frame_{frame_idx}_optimized_interp_{new_n}slice.nii.gz"
        )
        sitk.WriteImage(pred_itk_3d, save_path_3d)
        print(f"Saved 3D optimized interpolated volume (총 {new_n} slice) → {save_path_3d}\n")
