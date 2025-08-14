import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# --- 0) Volume 로드 & Slice 인덱스 설정 ---
cine = nib.load("/workspace/storage/kjh/cardiac/DMCVR/test/cropped_sax_np.nii.gz")
vol3d = cine.get_fdata()           # shape = (H, W, Z)

lower_idx, upper_idx, mid_idx = 4, 6, 5

# --- 1) 98th percentile clipping + [0,1] 정규화 함수 ---
def clip98_normalize(x):
    mx = np.percentile(x, 98)
    return np.clip(x, 0, mx) / mx

# --- 2) tanh([-1,1]) → [0,1] convert only for DiffAE outputs ---
def tanh_to_01(x):
    x01 = (x + 1.0) / 2.0
    return np.clip(x01, 0.0, 1.0)

# 2-1) GT 전처리 (3D → 2D slice)
vol_nrm3d = clip98_normalize(vol3d)
gt_nrm    = vol_nrm3d[:, :, mid_idx]      # (H, W)

# 2-2) Linear / Nearest raw slices
lo_nrm = vol_nrm3d[:, :, lower_idx]
hi_nrm = vol_nrm3d[:, :, upper_idx]


# 2-3) 외부 예측 로드 (tanh 출력 가정)

# 1-3) 외부 예측 로드 (이미 [0,1]이라 가정)
diffae_pred     = np.load("/workspace/storage/kjh/cardiac/DMCVR/test/output/4-6/pred_init_img.npy")
diffae_lax_pred = np.load("/workspace/storage/kjh/cardiac/DMCVR/test/output/4-6/final_pred.npy")

# --- 3) 모델별 2D slice 전처리 & 정규화 ---
models_nrm = {
    "Linear":     0.5*(lo_nrm + hi_nrm),        # already in [0,1]
    "Nearest":    lo_nrm,                       # already in [0,1]
    "DiffAE":     tanh_to_01(diffae_pred),      # [-1,1]→[0,1]
    "DiffAE_LAX": tanh_to_01(diffae_lax_pred),  # [-1,1]→[0,1]
}

# --- 4) 8-bit 변환 (PSNR/SSIM용) ---
gt_u8 = (gt_nrm * 255).round().astype(np.uint8)
models_u8 = {
    name: (pred * 255).round().astype(np.uint8)
    for name, pred in models_nrm.items()
}

# --- 5) LPIPS 초기화 ---
lpips_fn = lpips.LPIPS(net='alex').cuda()
def compute_lpips_uint8(a_u8, b_u8):
    t0 = torch.from_numpy(a_u8[None,None]/127.5 - 1).float().cuda()
    t1 = torch.from_numpy(b_u8[None,None]/127.5 - 1).float().cuda()
    with torch.no_grad():
        return lpips_fn(t0, t1).item()

# --- 6) 평가(metric 계산) ---
metrics = {}
for name, pred_f in models_nrm.items():
    pred_u = models_u8[name]
    
    ### modified start ###
    
    # psnr = peak_signal_noise_ratio(gt_u8, pred_u, data_range=255)
    # ssim = structural_similarity(gt_u8, pred_u, data_range=255)
    psnr = peak_signal_noise_ratio(gt_u8, np.flip(np.flip(pred_u, axis=0),axis=1), data_range=255)
    ssim = structural_similarity(gt_u8, np.flip(np.flip(pred_u, axis=0),axis=1), data_range=255)

    ### modified end ###

    rmse = np.sqrt(np.mean((gt_nrm - pred_f)**2))
    lp   = compute_lpips_uint8(gt_u8, pred_u)
    metrics[name] = dict(PSNR=psnr, SSIM=ssim, RMSE=rmse, LPIPS=lp)
    print(f"[{name:10}] PSNR={psnr:.2f}dB  SSIM={ssim:.3f}  RMSE={rmse:.4f}  LPIPS={lp:.4f}")

# --- 7) 시각화 및 저장 ---
output_dir = "/workspace/storage/kjh/cardiac/DMCVR/test/output/figures"
os.makedirs(output_dir, exist_ok=True)

titles = ["GT"] + list(models_nrm.keys())
images = [gt_nrm] + [models_nrm[n] for n in models_nrm]

fig, axes = plt.subplots(1, len(images), figsize=(4*len(images), 4))
for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

for i, name in enumerate(titles[1:], start=1):
    m = metrics[name]
    txt = (
        f"PSNR={m['PSNR']:.1f}dB\n"
        f"SSIM={m['SSIM']:.3f}\n"
        f"RMSE={m['RMSE']:.4f}\n"
        f"LPIPS={m['LPIPS']:.4f}"
    )
    axes[i].text(0.5, -0.2, txt, ha="center", va="top",
                 transform=axes[i].transAxes, fontsize=10)

plt.tight_layout()
save_path = os.path.join(output_dir, "evaluation_comparison.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"[Done] Saved evaluation figure to: {save_path}")