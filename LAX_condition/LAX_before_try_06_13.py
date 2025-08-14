import os
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
from bitsandbytes.optim import AdamW8bit
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pytorch_msssim import ms_ssim
from math import acos, sin
from templates import *  # autoenc_base, LitModel, ModelName
import matplotlib.pyplot as plt 

def main():
    # -----------------------------
    # 1) Distributed setup
    # -----------------------------
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    assert world_size == 4, f"4 GPU 전용 스크립트입니다. 현재 world_size={world_size}"

    # -----------------------------
    # 2) Model & EMA load (no DDP)
    # -----------------------------
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
        '/storage/kjh/cardiac/DMCVR/generation/diffae_multi_PNU/'
        'checkpoints_multi/med256_autoenc/1/epoch=438-step=756789.ckpt',
        map_location="cpu"
    )
    base_model.load_state_dict(ckpt["state_dict"], strict=True)
    base_model.to(device).eval()
    ema = base_model.ema_model.to(device).eval()
    model = ema  # inference model

    # -----------------------------
    # 3) Data preparation
    # -----------------------------
    file_path = "/storage/kjh/dataset/cardiac/PNU_cardiac/CINE/" \
                "test/middle_slice/sax/MR_Heart_834_crop_sa.nii.gz"
    filling   = 8
    t         = 30
    num_epoch = 50
    chunk_size = 20

    # load 3D volume → (Z,H,W)
    img_itk = sitk.ReadImage(file_path)
    vol_np  = sitk.GetArrayFromImage(img_itk).astype(np.float32)
    mx      = np.percentile(vol_np, 98)
    vol_np  = np.clip(vol_np, 0, mx) / mx * 2.0 - 1.0
    vol_np  = np.flip(vol_np, axis=-2).copy()[0]  # frame=0

    nonzero_idx = [i for i in range(len(vol_np)) if vol_np[i].max() != 0]
    vol_nz      = vol_np[nonzero_idx]

    # GT loads
    base_gt   = "/storage/icml_data_collection/Cardiac_database/" \
                "PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_834/resample_lax/" \
                "{ch}ch_nii/{ch}ch_{frame}.nii.gz"
    base_mask = "/storage/icml_data_collection/Cardiac_database/" \
                "PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_834/reg/" \
                "mask_reg_{ch}ch/mask_reg_{ch}ch_{frame}.nii.gz"

    frame      = 0
    gt_volumes = []
    mask_mids  = []
    for ch in [2,3,4]:
        gt = sitk.GetArrayFromImage(sitk.ReadImage(base_gt.format(ch=ch, frame=frame)))
        gt = np.clip(gt, 0, np.nanpercentile(gt, 98)) / np.nanpercentile(gt, 98)
        gt = np.flip(gt, axis=1).astype(np.float32)
        mid = gt.shape[0] // 2
        # negative-stride 방지용 copy()
        gt_mid = gt[mid].copy()
        gt_volumes.append(
            torch.from_numpy(gt_mid).unsqueeze(0).unsqueeze(0).to(device)
        )
        m = sitk.GetArrayFromImage(
            sitk.ReadImage(base_mask.format(ch=ch, frame=frame))
        ).astype(bool)
        m = np.flip(m, axis=1).copy()
        mask_mids.append(m[mid])  # (H,W)

    # build sampling grids
    grids_2d = []
    for mask_mid in mask_mids:
        H, W = mask_mid.shape
        ys, xs = np.where(mask_mid)
        x_ndc = 2*(xs/(W-1)) - 1
        y_ndc = 2*(ys/(H-1)) - 1
        coords = np.stack([x_ndc, y_ndc], axis=1)
        grids_2d.append(torch.from_numpy(coords).float().to(device).view(1,-1,1,2))

    I, J = np.meshgrid(
        np.arange(mask_mids[0].shape[0]),
        np.arange(mask_mids[0].shape[1]),
        indexing='ij'
    )
    full2d = torch.from_numpy(
        np.stack([2*(J/J.max())-1, 2*(I/I.max())-1], axis=-1)
    ).float().unsqueeze(0).to(device)

    # encode + stochastic encode
    img_tensor  = torch.from_numpy(vol_nz).float().unsqueeze(1).to(device)
    cond = base_model.encode(img_tensor)
    xTs  = base_model.encode_stochastic(img_tensor, cond, T=t)

    # spherical + linear interpolation
    def slerp_np(x0,x1,a):
        theta = acos(np.dot(x0.flatten(),x1.flatten())/
                     (np.linalg.norm(x0)*np.linalg.norm(x1)))
        return (sin((1-a)*theta)*x0+sin(a*theta)*x1)/sin(theta)

    final_x, final_c = [], []
    xTs_np = xTs.cpu().numpy(); c_np = cond.cpu().numpy()
    for i in range(len(xTs_np)-1):
        final_x.append(xTs_np[i]); final_c.append(c_np[i])
        for a in np.linspace(0,1,filling,endpoint=False)[:-1]:
            final_x.append(slerp_np(xTs_np[i],xTs_np[i+1],a))
            final_c.append((1-a)*c_np[i]+a*c_np[i+1])
    final_x.append(xTs_np[-1]); final_c.append(c_np[-1])

    interp_xTs   = torch.from_numpy(np.stack(final_x)).float()
    interp_conds = torch.from_numpy(np.stack(final_c)).float()
    x_t_gpu  = interp_xTs.to(device).requires_grad_(True)
    cond_gpu = interp_conds.to(device).requires_grad_(True)

    # optimizer & scheduler
    scaler    = GradScaler()
    optimizer = AdamW8bit([x_t_gpu,cond_gpu],lr=1e-2,weight_decay=1e-2)
    scheduler = StepLR(optimizer,step_size=50,gamma=0.5)
    total = x_t_gpu.shape[0]
    indices = list(range(local_rank,total,world_size))

    # -----------------------------
    # 5) Training Loop
    # -----------------------------
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        epoch_loss = 0.0

        for i in range(0,len(indices),chunk_size):
            batch_idx = indices[i:i+chunk_size]
            xs, cs = x_t_gpu[batch_idx], cond_gpu[batch_idx]

            # 1) render & debug print
            with autocast():
                chunk = checkpoint(base_model.render, xs, cs, t)

            # debug: 첫 슬라이스 상태
            sp0 = chunk[0]
            print(f"[DEBUG] after render: min={sp0.min():.4f}, max={sp0.max():.4f}, nan={torch.isnan(sp0).any()}")

            # optional AMP off test:
            # chunk = checkpoint(base_model.render, xs.float(), cs.float(), t)

            chunk_loss = torch.tensor(0.0,device=device)

            for sp in chunk:
                # ROI NCC + full SSIM
                for g2, gt5 in zip(grids_2d, gt_volumes):
                    pts_pred = F.grid_sample(sp.unsqueeze(0), g2, mode='bilinear', align_corners=True)
                    pts_gt   = F.grid_sample(gt5,   g2, mode='bilinear', align_corners=True)
                    ux, uy = pts_pred.mean(), pts_gt.mean()
                    vx = ((pts_pred-ux)**2).mean()
                    vy = ((pts_gt-uy)**2).mean()
                    vxy = ((pts_pred-ux)*(pts_gt-uy)).mean()
                    ncc_l = -vxy/(torch.sqrt(vx*vy)+1e-5)
                    ncc_l = torch.nan_to_num(ncc_l, nan=0.0)

                    # full-plane SSIM
                    ip = sp.unsqueeze(1)
                    ip = F.grid_sample(ip, full2d, mode='bilinear', align_corners=True).squeeze(1)
                    ig = gt5
                    ig = F.grid_sample(ig, full2d, mode='bilinear', align_corners=True).squeeze(1)

                    # scale to [0,1]
                    ip = (ip+1)*0.5; ig = ig.clamp(0,1)
                    ssim_l = 1 - ms_ssim(ip.unsqueeze(1), ig.unsqueeze(1),
                                        data_range=1.0, win_size=7)
                    ssim_l = torch.nan_to_num(ssim_l,nan=0.0)

                    chunk_loss += 0.5*ncc_l + 1.0*ssim_l

            # debug: loss before backward
            print(f"[DEBUG] chunk_loss={chunk_loss.item():.6f}")

            scaler.scale(chunk_loss).backward()
            epoch_loss += chunk_loss.item()
            del chunk, xs, cs
            torch.cuda.empty_cache()

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # sync & print
        lt = torch.tensor(epoch_loss,device=device)
        dist.all_reduce(lt,op=dist.ReduceOp.SUM)
        if local_rank==0:
            print(f"[Epoch {epoch+1}/{num_epoch}] Loss={lt.item()/world_size:.6f}")

    dist.barrier()
    if local_rank == 0:
        # -----------------------------
        # 0) frame_idx, sid 정의
        # -----------------------------
        sid = os.path.basename(file_path).split("_")[2]  # e.g. "834"
        frame_idx = frame  # 위에서 frame = 0 으로 정의한 변수

        base_model.eval()

        # --- 4D 메타 가져오기 ---
        orig_spacing_4d   = img_itk.GetSpacing()    # (sx, sy, sz, st)
        orig_origin_4d    = img_itk.GetOrigin()     # (ox, oy, oz, ot)
        orig_direction_4d = img_itk.GetDirection()  # len=16

        # --- 3D용으로 슬라이스 ---
        spacing3d  = orig_spacing_4d[:3]
        origin3d   = orig_origin_4d[:3]
        if len(orig_direction_4d) == 16:
            # 4×4 방향행렬에서 앞 3×3 부분만
            direction3d = (
                orig_direction_4d[0], orig_direction_4d[1], orig_direction_4d[2],
                orig_direction_4d[4], orig_direction_4d[5], orig_direction_4d[6],
                orig_direction_4d[8], orig_direction_4d[9], orig_direction_4d[10],
            )
        else:
            direction3d = orig_direction_4d  # 이미 9개라면 그대로

        orig_n = vol_nz.shape[0]  # 최적화 전 슬라이스 수

        # =========================================
        # 1) DiffAE로 predict한 데이터 저장 (before optimization)
        # =========================================
        with torch.no_grad():
            pred_before = base_model.render(
                interp_xTs.to(device), interp_conds.to(device), t
            )  # (D,1,H,W)

        vol_before = pred_before.squeeze(1).cpu().numpy().astype(np.float32)  # (D,H,W)
        vol_before = np.flip(vol_before, axis=1).copy()  # y축 뒤집기
        # vol_before = np.clip(vol_before, 0.0, 1.0)

        new_n     = vol_before.shape[0]
        new_z     = spacing3d[2] * (orig_n - 1) / (new_n - 1)

        itk_before = sitk.GetImageFromArray(vol_before)
        itk_before.SetSpacing((spacing3d[0], spacing3d[1], new_z))
        itk_before.SetOrigin ((origin3d[0],  origin3d[1],  origin3d[2]))
        itk_before.SetDirection(direction3d)

        save_dir_b = f"/storage/kjh/dataset/cardiac/output/DiffAE/subject_{sid}/frame_{frame_idx}/"
        os.makedirs(save_dir_b, exist_ok=True)
        save_path_b = os.path.join(
            save_dir_b,
            f"MR_Heart_{sid}_frame_{frame_idx}_pred_before_{new_n}slice.nii.gz"
        )
        sitk.WriteImage(itk_before, save_path_b)
        print(f"[Rank {local_rank}] Saved pre-opt pred → {save_path_b}")

        # =========================================
        # 2) 최적화한 데이터 저장 (after optimization)
        # =========================================
        with torch.no_grad():
            pred_after = base_model.render(x_t_gpu, cond_gpu, t)  # (D,1,H,W)

        vol_after = pred_after.squeeze(1).cpu().numpy().astype(np.float32)
        vol_after = np.flip(vol_after, axis=1).copy()
        # vol_after = np.clip(vol_after, 0.0, 1.0)

        itk_after = sitk.GetImageFromArray(vol_after)
        itk_after.SetSpacing((spacing3d[0], spacing3d[1], new_z))
        itk_after.SetOrigin ((origin3d[0],  origin3d[1],  origin3d[2]))
        itk_after.SetDirection(direction3d)

        save_dir_a = f"/storage/kjh/dataset/cardiac/output/Inverse/subject_{sid}/frame_{frame_idx}/"
        os.makedirs(save_dir_a, exist_ok=True)
        save_path_a = os.path.join(
            save_dir_a,
            f"MR_Heart_{sid}_frame_{frame_idx}_optimized_{new_n}slice.nii.gz"
        )
        sitk.WriteImage(itk_after, save_path_a)
        print(f"[Rank {local_rank}] Saved post-opt volume → {save_path_a}")

    dist.destroy_process_group()


        
if __name__=="__main__":
    # torchrun --nproc_per_node=4 LA_condition_all_pred_after.py
    main()