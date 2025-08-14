import os
import sys
import glob
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
from bitsandbytes.optim import AdamW8bit
import SimpleITK as sitk
from templates import *  # autoenc_base, LitModel, ModelName
import ants

def main():
    # 1) Distributed setup
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    assert world_size == 4, f"4 GPU 전용 스크립트입니다. 현재 world_size={world_size}"

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
        '/storage/kjh/cardiac/DMCVR/generation/diffae_multi_PNU/checkpoints_multi/med256_autoenc/1/epoch=438-step=756789.ckpt', map_location="cpu"
    )
    base_model.load_state_dict(ckpt["state_dict"], strict=True)
    base_model.to(device).eval()
    ema = base_model.ema_model.to(device).eval()
    model = ema

    # 3) Data preparation
    frame = 0  # if needed
    t =30
    num_epoch = 20
    chunk_size = 30
    lambda_roi    = 5.0  
    lambda_latent = 0.02   

    id_list = [
        148, 400, 769, 834, 243, 813, 600, 100, 708,
        342, 790, 823, 114, 760, 665, 812, 788, 515, 810,
        814, 786, 835, 182, 410, 724, 526, 704, 723, 387,
        776, 488, 711, 482, 190, 648, 692, 722, 664, 599
    ]

    # 2. 전처리 함수 (bias + clipping)
    def preprocess_n4_clip(arr, low_pct=1, high_pct=98):
        img_ants = ants.from_numpy(arr)
        arr_n4   = ants.n4_bias_field_correction(img_ants).numpy()
        lo, hi   = np.percentile(arr_n4, [low_pct, high_pct])
        return np.clip(arr_n4, lo, hi)

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

    for id in id_list : 
        # -- SAX volume
        sax_path = f"/storage/kjh/dataset/cardiac/PNU_cardiac/CINE/test/middle_slice/sax/MR_Heart_{id}_crop_sa.nii.gz"
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
        gt_path   = f"/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{id}/combine_1lax_cropped/combine_1lax_cropped_{frame}.nii.gz"
        mask_path = f"/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{id}/combine_1lax_mask_cropped_adjusted/combine_1lax_mask_cropped_adjusted_{frame}.nii.gz"
        interp_sax_path = f"/storage/icml_data_collection/Cardiac_database/PNUH/PNUH_cine_LAX_SAX_align_2/MR_Heart_{id}/sax_resample_linear_cropped/sax_resample_linear_cropped_{frame}.nii.gz"

        gt_vol = sitk.GetArrayFromImage(sitk.ReadImage(gt_path.format(frame=frame))).astype(np.float32)
        gt_vol = preprocess_n4_clip(gt_vol)
        gt_vol = np.clip(gt_vol, 0, np.nanpercentile(gt_vol, 98)) / np.nanpercentile(gt_vol, 98)
        gt_vol = np.flip(gt_vol, axis=1).copy()

        mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(mask_path.format(frame=frame))).astype(bool)
        mask_vol = np.flip(mask_vol, axis=1).copy()


        sax_idx, _, _ = vol_sax.shape


        linear_sax = inter_sax_load(interp_sax_path.format(frame=frame))

        # B) i*8번째 slice에서만 분포 추출 → quantile map
        sax_vals, comb_vals = [], []
        for z in range(sax_idx):
            sax_vals .append(linear_sax[:, :, z*8][mask_vol[:, :, z*8]>0])
            comb_vals.append(gt_vol[:, :, z*8][mask_vol[:, :, z*8]>0])
        src_q, tgt_q = make_quantile_map(np.concatenate(comb_vals),
                                        np.concatenate(sax_vals), n_quant=101)
        # C) combine_data 전체 정규화
        flat_in      = gt_vol.flatten()
        gt_vol_norm = np.interp(flat_in, src_q, tgt_q).reshape(gt_vol.shape)

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
        my_interp_xTs = torch.nn.Parameter(
            before_interp_xTs[idxs].clone().detach().to(device),
            requires_grad=True
        )
        my_interp_cs = torch.nn.Parameter(
            before_interp_cs[idxs].clone().detach().to(device),
            requires_grad=True
        )

        # 초기값 저장 (prior 계산용)
        my_interp_xTs_init = my_interp_xTs.data.clone()
        my_interp_cs_init  = my_interp_cs.data.clone()

        # --- 5) Optimizer, Scheduler, AMP 설정 ---
        optimizer = AdamW8bit([my_interp_xTs, my_interp_cs], lr=1e-2, weight_decay=1e-2)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        scaler    = GradScaler()

        # --- 6) Training Loop (전체 loss) ---
        for epoch in range(num_epoch):
            optimizer.zero_grad()
            epoch_loss = 0.0

            for i in range(0, len(idxs), chunk_size):
                batch_slice = slice(i, i + chunk_size)
                xs = my_interp_xTs[batch_slice]
                cs = my_interp_cs[batch_slice]
                batch_idx = idxs[batch_slice]

                with autocast():
                    pred = base_model.render( xs, cs, t)  # (chunk,1,H,W)
                    chunk = pred["sample"]
                    chunk = ( chunk +1 ) / 2 
                # print(chunk.requires_grad)          # → False
                # print(chunk.grad_fn)                # → None
                # print(xs.requires_grad)             # → True

                # 볼륨 & ROI 구성
                chunk_vol = chunk.permute(1, 0, 2, 3).unsqueeze(0)
                gt_vol    = gt_tensor[batch_idx].permute(1,0,2,3).unsqueeze(0)
                mask_vol  = mask_tensor[batch_idx].permute(1,0,2,3).unsqueeze(0)

                n_valid = mask_vol.sum().float()

                chunk_roi = chunk_vol * mask_vol
                gt_roi    = gt_vol    * mask_vol

                mu_gt   = gt_roi.sum() / n_valid
                sigma_gt= torch.sqrt(((gt_roi - mu_gt) * mask_vol).pow(2).sum() / n_valid + 1e-6)

                # 2) ROI 통계 및 Affine normalization
                mu_pred  = chunk_roi.sum() / n_valid
                sigma_pred = torch.sqrt(
                    ((chunk_roi - mu_pred) * mask_vol).pow(2).sum() / n_valid + 1e-3
                )
                sigma_pred = torch.clamp(sigma_pred, min=1e-3)
                pred_norm     = (chunk_roi - mu_pred) * (sigma_gt / sigma_pred) + mu_gt
                pred_norm_roi = chunk_roi * mask_vol

                # print("n_valid:", n_valid)
                # print("pred_norm_roi.sum", pred_norm_roi.sum().float())
                # print("gt_roi.sum:", gt_roi.sum().float())
                # print("mu_pred:", mu_pred)
                # print("mu_gt:", mu_gt)

                # 1) ROI-only MSE loss
                roi_loss = F.mse_loss(pred_norm_roi, gt_roi, reduction='sum') / n_valid
                

                # # 1) ROI-only MSE loss
                # pred_vals = chunk_roi[mask_vol.bool()]
                # gt_vals   = gt_roi[mask_vol.bool()]
                # roi_loss  = F.mse_loss(pred_vals, gt_vals, reduction='sum') / mask_vol.sum()  # ^[1]

                # 2) Latent prior loss
                latent_reg = ((xs - my_interp_xTs_init[batch_slice])**2).mean()
                cond_reg   = ((cs - my_interp_cs_init[batch_slice])**2).mean()
                prior_loss = lambda_latent * (latent_reg + cond_reg)

                # 3) Total loss
                chunk_loss = lambda_roi * roi_loss + prior_loss

                print(f"[DEBUG] roi_loss={roi_loss.item():.6f}, prior={prior_loss.item():.6f}")

                # backward & accumulate
                scaler.scale(chunk_loss).backward()
                epoch_loss += chunk_loss.item()

                # cleanup
                del chunk, xs, cs
                torch.cuda.empty_cache()

            # optimizer update
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Distributed loss sync & 출력
            lt = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(lt, op=dist.ReduceOp.SUM)
            if local_rank == 0:
                avg_loss = lt.item() / world_size
                print(f"[Epoch {epoch+1}/{num_epoch}] Loss={avg_loss:.6f}")

        print("Optimization 완료")
        # === 최적화된 slice 저장 ===

        # 메타 추출
        sid = os.path.basename(sax_path).split("_")[2]  # e.g. "834"
        frame_idx = frame

        orig_spacing_4d   = img_sax.GetSpacing()    # (sx, sy, sz, st)
        orig_origin_4d    = img_sax.GetOrigin()     # (ox, oy, oz, ot)
        orig_direction_4d = img_sax.GetDirection()  # len=16

        # --- 3D용으로 슬라이스 ---
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

        # 각 rank별 최적화 결과 준비
        local_xTs = my_interp_xTs.detach()
        local_cs  = my_interp_cs.detach()

        # shape 수집
        local_shapes = torch.tensor([local_xTs.shape[0]], device=device)
        all_shapes   = [torch.zeros_like(local_shapes) for _ in range(world_size)]

        # 1) barrier #1
        print(f"[Rank {local_rank}] barrier #1 before")
        dist.barrier()
        print(f"[Rank {local_rank}] barrier #1 after")

        # 2) all_gather #1: 각 rank별 slice 개수
        print(f"[Rank {local_rank}] all_gather #1 before")
        dist.all_gather(all_shapes, local_shapes)
        print(f"[Rank {local_rank}] all_gather #1 after")

        # shape 수집 후
        max_len = max(x.item() for x in all_shapes)

        # 1) xTs (4D) 패딩: 맨 앞 차원만 늘리기
        delta_x = max_len - local_xTs.shape[0]
        pad2_x  = torch.zeros(
            delta_x,
            *local_xTs.shape[1:],  # (C, H, W)
            device=device,
            dtype=local_xTs.dtype
        )
        pad_xTs = torch.cat([local_xTs, pad2_x], dim=0)

        # 2) cs (2D) 패딩: 맨 앞 차원만 늘리기
        delta_c = max_len - local_cs.shape[0]
        pad2_c  = torch.zeros(
            delta_c,
            local_cs.shape[1],      # latent_dim
            device=device,
            dtype=local_cs.dtype
        )
        pad_cs = torch.cat([local_cs, pad2_c], dim=0)

        print(f"[Rank {local_rank}]", pad_xTs.shape, pad_cs.shape)


        # 3) barrier #2
        print(f"[Rank {local_rank}] barrier #2 before")
        dist.barrier()
        print(f"[Rank {local_rank}] barrier #2 after")

        # 4) all_gather #2: padded tensor 수집
        gathered_xTs = [torch.zeros_like(pad_xTs) for _ in range(world_size)]
        gathered_cs  = [torch.zeros_like(pad_cs)  for _ in range(world_size)]
        print(f"[Rank {local_rank}] all_gather #2 before")
        dist.all_gather(gathered_xTs, pad_xTs)
        dist.all_gather(gathered_cs,  pad_cs)
        print(f"[Rank {local_rank}] all_gather #2 after")

        # 5) barrier #3
        print(f"[Rank {local_rank}] barrier #3 before")
        dist.barrier()
        print(f"[Rank {local_rank}] barrier #3 after")

        # 6) 결과 합치고 저장 (rank 0)
        if local_rank == 0:
            ns     = [int(x.item()) for x in all_shapes]
            full_x = torch.cat([g[:n] for g, n in zip(gathered_xTs, ns)], dim=0)
            full_c = torch.cat([g[:n] for g, n in zip(gathered_cs,  ns)], dim=0)
            with torch.no_grad():
                pred_op = base_model.render(full_x.to(device), full_c.to(device), t)
                pred_after = pred_op["sample"]
                pred_after = (pred_after + 1) / 2
            vol_after = pred_after.squeeze(1).cpu().numpy().astype(np.float32)
            vol_after = np.flip(vol_after, axis=1).copy()

            new_n = vol_after.shape[0]
            new_z = spacing3d[2] * (orig_n - 1) / (new_n - 1)

            # # --- 원본 SAX volume도 동일 spacing/new_z 로 저장 ---
            # itk_orig = sitk.GetImageFromArray(vol_sax.astype(np.float32))
            # itk_orig.SetSpacing((spacing3d[0], spacing3d[1], new_z))
            # itk_orig.SetOrigin((origin3d[0], origin3d[1], origin3d[2]))
            # itk_orig.SetDirection(direction3d)
            # save_dir_o = f"/storage/kjh/dataset/cardiac/output/DiffAE/subject_{sid}/frame_{frame_idx}/"
            # os.makedirs(save_dir_o, exist_ok=True)
            # save_path_o = os.path.join(
            #     save_dir_o,
            #     f"MR_Heart_{sid}_frame_{frame_idx}_original_{orig_n}slice.nii.gz"
            # )
            # sitk.WriteImage(itk_orig, save_path_o)
            # print(f"[Rank {local_rank}] Saved original SAX volume → {save_path_o}")

            # === 최적화된 slice 저장 ===
            itk_after = sitk.GetImageFromArray(vol_after)
            itk_after.SetSpacing((spacing3d[0], spacing3d[1], new_z))
            itk_after.SetOrigin((origin3d[0], origin3d[1], origin3d[2]))
            itk_after.SetDirection(direction3d)
            save_dir_a  = f"/storage/kjh/dataset/cardiac/output/Inverse/subject_{sid}/frame_{frame_idx}/"
            os.makedirs(save_dir_a, exist_ok=True)
            save_path_a = os.path.join(
                save_dir_a,
                f"MR_Heart_{sid}_frame_{frame_idx}_optimized_{new_n}slice.nii.gz"
            )
            sitk.WriteImage(itk_after, save_path_a)
            print(f"[Rank {local_rank}] Saved post-opt volume → {save_path_a}")

        # 7) barrier #4: 다음 subject 준비
        print(f"[Rank {local_rank}] barrier #4 before")
        dist.barrier()
        print(f"[Rank {local_rank}] barrier #4 after")



    # 전체 subject 처리 후
    dist.destroy_process_group()

if __name__=="__main__":
    # torchrun --nproc_per_node=4 LA_condition_all_pred_after.py
    main()