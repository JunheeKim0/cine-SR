from templates import *
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from tqdm import tqdm
from math import acos, sin
import matplotlib.pyplot as plt

# Config of the generating parameters
skipping    = 0    # 0 = no skipping, 1 = skip every other slice
filling     = 30   # number of interpolated slices between each pair
t           = 30   # diffusion timestep
device      = 'cuda:0'

# Model setup
conf = autoenc_base()
conf.img_size             = 128
conf.net_ch               = 128
conf.net_ch_mult          = (1, 1, 2, 3, 4)
conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
conf.model_name           = ModelName.beatgans_autoenc
conf.name                 = 'med256_autoenc'
conf.make_model_conf()

model = LitModel(conf)
ckpt = torch.load(
    '/workspace/storage/kjh/cardiac/DMCVR/generation/diffae_multi/checkpoints_multi/'
    'med256_autoenc_400/epoch=262-step=676500.ckpt',
    map_location=device
)
model.load_state_dict(ckpt['state_dict'], strict=True)
model.ema_model.eval()
model.ema_model.to(device)

save_to_root = "/workspace/storage/kjh/cardiac/DMCVR/test/output"

# Data paths
base_path = "/workspace/storage/kjh/cardiac/DMCVR"
files     = glob.glob(os.path.join(base_path, "test", "cropped_sax_4d.nii.gz"))

# Linear interpolation for cond (z_sem)
def lin_interpolate(a, b, num=filling):
    alpha = 1.0/(num-1)
    return [(1 - i*alpha)*a + i*alpha*b for i in range(num)]

# Slerp interpolation for xT
def slerp_np(x0, x1, alpha):
    theta = acos(np.dot(x0.flatten(), x1.flatten()) /
                  (np.linalg.norm(x0)*np.linalg.norm(x1)))
    return (sin((1-alpha)*theta)*x0 + sin(alpha*theta)*x1) / sin(theta)

def slerp_interpolate(x0, x1, num=filling):
    alpha = 1.0/(num-1)
    return [slerp_np(x0, x1, alpha*i) for i in range(num)]

# 1) Fully differentiable decoder via model.render (no_grad removed)
def decode_det(z_sem: torch.Tensor, xT: torch.Tensor, t: int) -> torch.Tensor:
    z_b  = z_sem.unsqueeze(0).to(device)   # (1,512)
    xT_b = xT.unsqueeze(0).to(device)      # (1,128)
    # conditional render now gradient-enabled
    x0_pred = model.render(xT_b, cond=z_b, T=t)  # (1, 1, 128, 128)
    return x0_pred

# 2) Memory-efficient geodesic interpolation
def geodesic_interpolate(zA: torch.Tensor, zB: torch.Tensor, xT: torch.Tensor,
                         decoder_fn, steps=filling, lr=1e-2, iter_num: int=5):
    # initialize linear path
    zs = torch.stack([(1 - i/(steps-1))*zA + (i/(steps-1))*zB
                      for i in range(steps)], dim=0)
    zs = torch.nn.Parameter(zs)  # requires_grad=True
    optimizer = torch.optim.SGD([zs], lr=lr)

    for _ in range(iter_num):
        optimizer.zero_grad()
        # accumulate gradients per-pair and free graph immediately
        for k in range(steps-1):
            xk  = decoder_fn(zs[k],   xT, t)
            xk1 = decoder_fn(zs[k+1], xT, t)
            loss_k = F.mse_loss(xk, xk1)
            loss_k.backward()   # frees intermediate activations after backward
        optimizer.step()
        torch.cuda.empty_cache()  # release any cached memory

    return [zs_i.detach() for zs_i in zs]

# 3) Main interpolation loop
for src_path in tqdm(files, desc="Generating"):
    img_itk = sitk.ReadImage(src_path)
    vol_np  = sitk.GetArrayFromImage(img_itk)  # (Z, H, W)
    max98   = np.percentile(vol_np, 98)
    vol_np  = np.clip(vol_np, 0, max98) / max98
    vol_np  = vol_np*2 - 1

    img_np = vol_np[0]  # use first frame

    filt_idx = [i for i in range(len(img_np)) if img_np[i].max() != 0]
    filt_vol = img_np[filt_idx]

    tensor_img   = torch.from_numpy(filt_vol).float()
    input_tensor = tensor_img.unsqueeze(1).to(device)  # (D',1,128,128)

    # encode semantic and stochastic latents
    cond_tensor = model.encode(input_tensor)              # (D',512)
    xTs         = model.encode_stochastic(input_tensor,
                                          cond_tensor, T=t)  # (D',128)

    final_xTs   = []
    final_conds = []

    for i in range(xTs.shape[0] - 1):
        # append original
        final_xTs.append(xTs[i].cpu().numpy())
        final_conds.append(cond_tensor[i].cpu().numpy())

        # 1) noise latent interpolation (Slerp)
        inter_xT = slerp_interpolate(
            xTs[i].cpu().numpy(),
            xTs[i+1].cpu().numpy(),
            num=filling
        )
        final_xTs.extend(inter_xT)

        # 2) semantic latent geodesic interpolation
        zA = cond_tensor[i]      # (512,)
        zB = cond_tensor[i+1]
        xT_i = xTs[i]            # (128,)
        inter_zs = geodesic_interpolate(
            zA, zB, xT_i,
            decoder_fn=decode_det,
            steps=filling, lr=1e-1, iter_num=5
        )
        final_conds.extend([z.cpu().numpy() for z in inter_zs])

    # append last slice
    final_xTs.append(xTs[-1].cpu().numpy())
    final_conds.append(cond_tensor[-1].cpu().numpy())

    # convert to tensors and render all at once
    interp_xTs   = torch.from_numpy(np.stack(final_xTs)).float().to(device)
    interp_conds = torch.from_numpy(np.stack(final_conds)).float().to(device)
    pred = model.render(interp_xTs, cond=interp_conds, T=t)  # (N,1,128,128)

    print("Generated batch shape:", pred.shape)

    # save NIfTI
    out_itk = sitk.GetImageFromArray(pred.squeeze(1).cpu().numpy())
    orig_sp = img_itk.GetSpacing()
    orig_n  = len(filt_vol)
    new_n   = (orig_n - 1)*filling + orig_n
    new_z   = orig_sp[2] * orig_n / new_n
    out_itk.SetSpacing((orig_sp[0], orig_sp[1], new_z))
    out_itk.SetOrigin(img_itk.GetOrigin())

    # convert 4D direction to 3D if needed
    orig_dir = img_itk.GetDirection()
    if len(orig_dir) == 16:
        dir_3d = (
            orig_dir[0], orig_dir[1], orig_dir[2],
            orig_dir[4], orig_dir[5], orig_dir[6],
            orig_dir[8], orig_dir[9], orig_dir[10]
        )
    else:
        dir_3d = orig_dir
    out_itk.SetDirection(dir_3d)

    save_path = os.path.join(save_to_root, "img",
                             f"{os.path.basename(src_path).split('.')[0]}_interp_30_.nii.gz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sitk.WriteImage(out_itk, save_path)
