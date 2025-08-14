# hyperparam_search.py

import os
import numpy as np
import torch
import SimpleITK as sitk
import ants
import ex6_inverse_opt
import optuna
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# -----------------------------
# 1) Projection 및 GT 로드 함수
# -----------------------------

frame = 0

def preprocess_n4_clip(arr, low_pct=1, high_pct=98):
    img_ants = ants.from_numpy(arr)
    arr_n4   = ants.n4_bias_field_correction(img_ants).numpy()
    lo, hi   = np.percentile(arr_n4, [low_pct, high_pct])
    return np.clip(arr_n4, lo, hi) / hi

def projection_torch(vol: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    vol : (D, H, W)   – 3D volume tensor  
    mask: (D, H, W)   – 3D binary/probability mask tensor  
    returns: (H, W)  – XY-plane mean projection
    """
    masked_vol = vol * mask  # ROI masking
    sum_vol    = masked_vol.sum(dim=1)                # (H, W)
    sum_mask   = mask.sum(dim=1).clamp(min=1e-6)      # (H, W)
    mean_roi   = sum_vol / sum_mask                   # (H, W)
    return mean_roi

def load_lax_mask(lax_path, mask_path, device):
    gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(lax_path.format(frame=frame))).astype(np.float32)
    gt_vol = preprocess_n4_clip(gt_vol, low_pct=1, high_pct=99.5)
    gt_vol = np.flip(gt_vol, axis=1).copy()

    mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(mask_path.format(frame=frame))).astype(bool)
    mask_vol = np.flip(mask_vol, axis=1).copy()

    gt_vol_t   = torch.from_numpy(gt_vol).float().to(device)
    mask_vol_t = torch.from_numpy(mask_vol).float().to(device)
    gt_vol_proj = projection_torch(gt_vol_t, mask_vol_t)
    return gt_vol_proj, mask_vol_t

# -----------------------------
# 2) GT 이미지 및 mask 로드
# -----------------------------
def load_all_gt(subject_id, frame, device):
    """
    MR_Heart_{subject_id}에 대해 2ch/3ch/4ch GT projection 및 mask 로드
    return: dict {'2ch':{'proj':Tensor,'mask':Tensor},...}
    """
    gt = {}
    channels = ['2ch', '3ch', '4ch']
    variants = ['aligned', 'flipped_aligned']
    for ch in channels: 
        for var in variants:
            fname_img_full  = f'{ch}_nii_{var}_1lax_cropped'
            fname_mask_full = f'{ch}_nii_{var}_1lax_mask_cropped_adjusted'  
            lax_path  = os.path.join(f"/workspace/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{subject_id}", fname_img_full, f'{ch}_{var}_1lax_cropped_{frame}.nii.gz')
            mask_path = os.path.join(f"/workspace/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{subject_id}", fname_mask_full, f'{ch}_{var}_1lax_mask_cropped_adjusted_{frame}.nii.gz')    
            if os.path.exists(lax_path) and os.path.exists(mask_path):
                # 확장자 제거
                proj, mask = load_lax_mask(lax_path, mask_path, device)
                gt[ch] = {'proj': proj, 'mask': mask}
    return gt


# -----------------------------
# 3) PSNR / SSIM 계산 함수
# -----------------------------
def eval_metrics(pred: torch.Tensor, gt: torch.Tensor):
    """
    pred, gt: torch.Tensor (H, W) in [0,1] range
    returns: (ssim: float, psnr: float)
    """
    p = pred.detach().cpu().numpy()
    g = gt.detach().cpu().numpy()
    # structural_similarity returns scalar for 2D
    s = structural_similarity(g, p, data_range=1.0)
    pnr = peak_signal_noise_ratio(g, p, data_range=1.0)
    return s, pnr


# -----------------------------
# 4) Objective 함수 (공통 사용)
# -----------------------------
def objective_params(params, gt, device, frame, subject_id):
    preds = ex6_inverse_opt.optimize_sax(
        subject_id=subject_id,
        frame=frame,
        device=device,
        lambda_roi=params['l_roi'],
        lambda_prior=params['l_prior'],
        lambda_smooth=params['l_smooth'],
        lambda_latent=params['l_latent'],
        alpha_proj=params['alpha_proj']
    )
    total_ssim = 0
    for ch in ['2ch', '3ch', '4ch']:
        s, _ = eval_metrics(preds[ch], gt[ch]['proj'])
        total_ssim += s
    return total_ssim

# -----------------------------
# 5) Dynamic Coordinate Descent
# -----------------------------
def dynamic_search(initial, gt, device, frame, subject_id):
    params = initial.copy()
    for name in params.keys():
        step = 2.0
        while step > 1.01:
            current_val = params[name]
            best_val = current_val
            base_score = objective_params(params, gt, device, frame, subject_id)
            # 상승/하강 시도
            for factor in [step, 1.0/step]:
                params[name] = current_val * factor
                score = objective_params(params, gt, device, frame, subject_id)
                print(f"Testing {name}={params[name]:.6f} -> SSIM={score:.6f}")
                if score > base_score:
                    base_score, best_val = score, params[name]
            params[name] = best_val
            if best_val == current_val:
                step = step ** 0.5
        print(f"Optimized {name} = {params[name]:.6f}")
    return params

# -----------------------------
# 6) Main
# -----------------------------
if __name__ == '__main__':
    device = torch.device('cuda:0')
    # id_list = [
    #     400, 769, 834, 243, 813, 600, 100, 708,
    #     342, 790, 823, 114, 760, 665, 812, 788, 515, 810,
    #     814, 786, 835, 182, 410, 724, 526, 704, 723, 387,
    #     776, 488, 711, 482, 190, 648, 692, 722, 664, 599
    # ]
    subject_id = 400
    # GT load once
    gt = load_all_gt(subject_id, frame, device)

    # 초기 파라미터
    initial = {
        'l_roi':     5.0,
        'l_prior':   1.0,
        'l_smooth':  3.0,
        'l_latent':  0.02,
        'alpha_proj':0.001
    }

    best_params = dynamic_search(
        initial, gt, device, frame, subject_id
    )
    print("Final optimized params:", best_params)
