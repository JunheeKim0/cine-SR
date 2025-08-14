from templates import *
from tqdm import tqdm
import torch
import os
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn, peak_signal_noise_ratio as psnr_fn
from skimage.segmentation import find_boundaries
from math import acos, sin

# --- 사용자 설정 ---
vol_path   = "/workspace/storage/kjh/cardiac/DMCVR/test/cropped_sax_np.nii.gz"          # shape = (T, Z, H, W)
gt_path    = "/workspace/storage/kjh/cardiac/DMCVR/test/combined_overlay.nii.gz"       # shape = (T, Z, H, W)
model_ckpt = '/workspace/storage/kjh/cardiac/DMCVR/generation/diffae_multi/checkpoints_multi/med256_autoenc/epoch=262-step=676500.ckpt'
filling    = 5       # 보간할 슬라이스 개수
t          = 100     # DDIM 노이즈 단계
opt_idx    = 2       # 보간된 슬라이스 중 최적화할 인덱스
# -----------------------

k =0
conf = autoenc_base()
conf.img_size = 128
conf.net_ch = 128
conf.net_ch_mult = (1, 1, 2, 3, 4)
conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
conf.model_name = ModelName.beatgans_autoenc
# conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
# conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
conf.eval_every_samples = 10_000_000
conf.eval_ema_every_samples = 10_000_000
conf.total_samples = 200_000_000
# conf.batch_size = 1

conf.name = 'med256_autoenc'
conf.make_model_conf()

# 1) 4D 볼륨 로드
vol_arr = sitk.GetArrayFromImage(sitk.ReadImage(vol_path))       # (T,Z,H,W) 또는 (Z,H,W)
gt_arr  = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))        # (T,Z,H,W) 또는 (Z,H,W)

# 2) 만약 3D 볼륨이라면 첫 차원에 time/frame 축을 추가
if vol_arr.ndim == 3:
    vol_arr = vol_arr[None, ...]      # → (1, Z, H, W)
    gt_arr  = gt_arr[None, ...]

num_frames = vol_arr.shape[0]

vol_nz_all    = vol_arr
region_gt_all = gt_arr

# 2) 디바이스 및 모델 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LightningModule 인 경우 load_from_checkpoint 사용 가능
model = LitModel(conf)
# breakpoint()
state = torch.load(f'/workspace/storage/kjh/cardiac/DMCVR/generation/diffae_multi/checkpoints_multi/med256_autoenc/epoch=262-step=676500.ckpt', map_location=device)
model.load_state_dict(state['state_dict'], strict=True)
model.ema_model.eval()
model.ema_model.to(device)

# 3) 보간 함수 정의
def lin_interpolate(s1, s2, num=filling):
    alpha = 1.0 / (num - 1.0)
    return [i*alpha*s2 + (1-alpha*i)*s1 for i in range(num-1)]

def slerp_np(x0, x1, alpha):
    theta = acos(np.dot(x0.flatten(), x1.flatten()) /
                 (np.linalg.norm(x0)*np.linalg.norm(x1)))
    return (sin((1-alpha)*theta)*x0 + sin(alpha*theta)*x1) / sin(theta)

def slerp_interpolate(s1, s2, num=filling):
    alpha = 1.0 / (num - 1)
    return [slerp_np(s1, s2, alpha*i) for i in range(num-1)]

# 4) 처리할 슬라이스 페어
sel_pairs = [(3,5), (5,7), (7,9)]

# 5) 하이퍼파라미터
num_epochs    = 200
num_steps     = 1
base_lr       = 2e-3
lambda_roi    = 5.0
lambda_latent = 0.02
alpha_proj    = 0.001

# 6) Frame 루프
for frame_idx in range(num_frames):
    print(f"\n=== Processing frame {frame_idx+1}/{num_frames} ===")
    
    # 7) NIfTI → NumPy → Intensity normalize
    vol_np = vol_nz_all[frame_idx]   # (Z, H, W)
    # 98th percentile clipping & 0–1 스케일
    mx = np.percentile(vol_np, 98)
    vol_np = (np.clip(vol_np, 0, mx) / mx).astype(np.float32)   # ← cast to float32

    gt_np  = region_gt_all[frame_idx]  # (Z, H, W)
    gt_np  = gt_np.astype(np.float32)  # ← cast to float32
    
    # 8) Pair 루프
    for start_idx, end_idx in sel_pairs:
        print(f"--- Slice pair {start_idx}-{end_idx} ---")
        
        # a) 슬라이스 선택 및 중간 인덱스
        sel_idx = [start_idx, end_idx]
        vol_sel = vol_np[sel_idx]               # (2, H, W)
        mid_idx = (start_idx + end_idx)//2
        region_gt = gt_np[mid_idx]              # (H, W)
        real_data = vol_np[mid_idx]             # (H, W)
        
        # b) 모델 인코딩
        vol_sel_f = vol_sel.astype(np.float32)
        b, H, W   = vol_sel_f.shape
        # (2, H, W) → (2, 1, H, W)
        img_tensor = torch.from_numpy(vol_sel_f).view(b, 1, H, W).to(device)
        cond_tensor = model.encode(img_tensor)
        xTs         = model.encode_stochastic(img_tensor, cond_tensor, T=t)
        
        # c) 보간(interpolation)
        xTs_np  = xTs.cpu().numpy();       cond_np = cond_tensor.cpu().numpy()
        final_x, final_c = [], []
        for i in range(len(xTs_np)-1):
            final_x.append(xTs_np[i]);   final_c.append(cond_np[i])
            final_x.extend(slerp_interpolate(xTs_np[i], xTs_np[i+1], num=filling))
            final_c.extend(lin_interpolate(cond_np[i],  cond_np[i+1], num=filling))
        final_x.append(xTs_np[-1]);       final_c.append(cond_np[-1])
        interp_xTs   = torch.from_numpy(np.stack(final_x)).float().to(device)
        interp_conds = torch.from_numpy(np.stack(final_c)).float().to(device)
        
        # d) 최적화용 텐서 준비
        x_t_slice       = interp_xTs[opt_idx:opt_idx+1].detach().clone().requires_grad_(True).to(device)
        cond_slice      = interp_conds[opt_idx:opt_idx+1].detach().clone().requires_grad_(True).to(device)
        x_t_slice_init  = x_t_slice.detach().clone()
        cond_slice_init = cond_slice.detach().clone()
        
        # e) GT & mask 준비
        gt_tensor  = torch.from_numpy(region_gt).unsqueeze(0).unsqueeze(0).to(device)
        gt_clean   = torch.nan_to_num(gt_tensor, nan=0.0)
        valid_mask = (~torch.isnan(gt_tensor)).float()
        n_valid    = valid_mask.sum()
        gt_roi     = gt_clean * valid_mask
        mu_gt      = gt_roi.sum() / n_valid
        sigma_gt   = torch.sqrt(((gt_roi-mu_gt)*valid_mask).pow(2).sum()/n_valid + 1e-6)
        
        # f) 옵티마이저 & 스케줄러
        optimizer = Adam([x_t_slice, cond_slice], lr=base_lr)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        
        # g) loss 기록용 리스트
        orig_ps, orig_ss = [], []
        opt_ps,  opt_ss  = [], []
        
        # h) 최적화 루프
        for epoch in range(num_epochs):
            for _ in range(num_steps):
                # 1) Forward
                pred = model.render(x_t_slice, cond=cond_slice, T=t)
                # 2) ROI 정규화
                p_roi      = pred * valid_mask
                mu_p       = p_roi.sum() / n_valid
                sigma_p    = torch.sqrt(((p_roi - mu_p) * valid_mask).pow(2).sum() / n_valid + 1e-6)
                p_norm     = (pred - mu_p) * (sigma_gt / sigma_p) + mu_gt
                p_norm_roi = p_norm * valid_mask
                # 3) Loss 계산
                roi_loss = F.mse_loss(p_norm_roi, gt_roi, reduction='sum') / n_valid
                lat_reg  = ((x_t_slice - x_t_slice_init) ** 2).mean()
                cnd_reg  = ((cond_slice  - cond_slice_init) ** 2).mean()
                prior_l  = lambda_latent * (lat_reg + cnd_reg)
                loss     = lambda_roi * roi_loss + prior_l

                # 4) Backward & Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 5) Projection
                with torch.no_grad():
                    x_t_slice.data  = (1 - alpha_proj) * x_t_slice.data  + alpha_proj * x_t_slice_init
                    cond_slice.data = (1 - alpha_proj) * cond_slice.data + alpha_proj * cond_slice_init

            # 6) Scheduler step
            scheduler.step()

        # 최적화 완료 후 PSNR·SSIM 계산 및 출력
        with torch.no_grad():
            init_pred = model.render(x_t_slice_init, cond=cond_slice_init, T=t).squeeze().cpu().numpy()
            opt_pred  = model.render(x_t_slice,      cond=cond_slice,      T=t).squeeze().cpu().numpy()

        dr = real_data.max() - real_data.min() + 1e-6
        init_psnr = psnr_fn(real_data, init_pred, data_range=dr)
        init_ssim = ssim_fn(real_data, init_pred, data_range=dr)
        opt_psnr  = psnr_fn(real_data, opt_pred,  data_range=dr)
        opt_ssim  = ssim_fn(real_data, opt_pred,  data_range=dr)

        print(f"Init  → PSNR: {init_psnr:.2f}, SSIM: {init_ssim:.4f}")
        print(f"Opt   → PSNR: {opt_psnr:.2f}, SSIM: {opt_ssim:.4f}")

        # 최종 시각화 저장 (비차단)
        # valid_mask를 bool 배열로 변환한 뒤 boundary 추출
        mask_np   = valid_mask.squeeze().cpu().numpy().astype(bool)
        mask_edge = find_boundaries(mask_np, mode='outer')

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(init_pred,  cmap='gray'); axs[0].set_title('Init Pred');      axs[0].axis('off')
        axs[1].imshow(real_data,  cmap='gray'); axs[1].set_title('GT Patch');      axs[1].axis('off')
        axs[2].imshow(opt_pred,   cmap='gray'); axs[2].set_title('Opt Pred');      axs[2].axis('off')
        axs[3].imshow(region_gt,  cmap='gray'); axs[3].contour(mask_edge, colors='r'); axs[3].axis('off')
        plt.tight_layout()

        viz_path = os.path.join('/workspace/storage/kjh/cardiac/DMCVR/test', f"viz_f{frame_idx}_p{start_idx}-{end_idx}.png")
        fig.savefig(viz_path)
        plt.close(fig)

        
print("\n모든 frame/pair 최적화 완료. 누락된 코드는 없습니다.")