from templates import *
import matplotlib.pyplot as plt
import numpy as np
from math import acos, sin
import SimpleITK as sitk
from tqdm import tqdm
import sys
import os

# ─── 0) GPU 설정 ───────────────────────────────────────────────────────────
# 사용할 GPU 인덱스를 지정합니다 (예: 0,1,2,3)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 1) 모델 로드 & DataParallel 래핑 ────────────────────────────────────
conf = autoenc_base()
conf.img_size                = 128
conf.net_ch                  = 128
conf.net_ch_mult             = (1, 1, 2, 3, 4)
conf.net_enc_channel_mult    = (1, 1, 2, 3, 4, 4)
conf.model_name              = ModelName.beatgans_autoenc
conf.eval_every_samples      = 10_000_000
conf.eval_ema_every_samples  = 10_000_000
conf.total_samples           = 200_000_000
conf.name                    = 'med256_autoenc'
conf.make_model_conf()

# LightningModule 생성 및 체크포인트 로드
base_model = LitModel(conf)
ckpt_path  = '/storage/kjh/cardiac/DMCVR/generation/diffae_multi_PNU/checkpoints_multi/med256_autoenc/epoch=295-step=708500.ckpt'
state      = torch.load(ckpt_path, map_location='cpu', weights_only=True)
base_model.load_state_dict(state['state_dict'], strict=True)
base_model.eval()

# base_model 전체를 DataParallel로 감싸 멀티 GPU 사용 준비
model = nn.DataParallel(base_model.to(device))

# ─── 2) 입력/출력 경로 설정 ────────────────────────────────────────────────
base_path    = "/storage/kjh/dataset/cardiac/PNU_cardiac/CINE/test/middle_slice"

# 처리할 NIfTI 파일 리스트 (여기서는 특정 파일만 예시로 불러옴)
files = glob.glob(os.path.join(base_path, "sax", "*_crop_sa.nii.gz"))
# files = glob.glob(os.path.join(base_path, "sax", "MR_Heart_664_crop_sa.nii.gz"))
print("Files to process:", files)

# ─── 3) 보간 파라미터 ─────────────────────────────────────────────────────
skipping = 0   # 현재 스크립트에서 사용되진 않음
filling  = 8   # 각 인접 슬라이스 사이에 삽입할 보간 개수
t        = 30  # diffusion T 스텝
f        = 0   # 관심 frame index (여기서는 0 고정)

# 선형 보간 l_sem
def lin_interpolate(slice_1,slice_2,num=filling):
    #num is how many slices need to be inserted
    alpha=1.0/(num-1.0)
    out=[]
    # out.append(slice_1)
    for i in range(num-1):
        out.append((i)*alpha*slice_2+(1.0-(i)*alpha)*slice_1)
    # out.append(slice_2)
    return out

def slerp(x0: torch.Tensor, x1: torch.Tensor, alpha: float) -> torch.Tensor:
    """Spherical Linear intERPolation
    Args:
        x0 (`torch.Tensor`): first tensor to interpolate between
        x1 (`torch.Tensor`): seconds tensor to interpolate between
        alpha (`float`): interpolation between 0 and 1
    Returns:
        `torch.Tensor`: interpolated tensor
    """

    theta = acos(torch.dot(torch.flatten(x0), torch.flatten(x1)) / torch.norm(x0) / torch.norm(x1))
    return sin((1 - alpha) * theta) * x0 / sin(theta) + sin(alpha * theta) * x1 / sin(theta)

def slerp_np(x0: np.ndarray, x1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical Linear intERPolation
    Args:
        x0 (`torch.Tensor`): first tensor to interpolate between
        x1 (`torch.Tensor`): seconds tensor to interpolate between
        alpha (`float`): interpolation between 0 and 1
    Returns:
        `torch.Tensor`: interpolated tensor
    """

    theta = acos(np.dot(x0.flatten(), x1.flatten()) / np.linalg.norm(x0) / np.linalg.norm(x1))
    return sin((1 - alpha) * theta) * x0 / sin(theta) + sin(alpha * theta) * x1 / sin(theta)

# 구형 보간 - X_t
def slerp_interpolate(slice_1,slice_2,num=filling):
    #num is how many slices need to be inserted
    alpha=1.0/(num-1)
    out=[]
    # out.append(slice_1)
    for i in range(num-1):
        out.append(slerp_np(slice_1,slice_2,alpha*i))
    # out.append(slice_2)
    return out

for f in range(25):
    # interpolation
    for src_path in tqdm(files, desc="Generating"):
        # 입력 경로 대비 상대경로로 출력 경로 구성
        img_id = os.path.basename(src_path).split(".")[0]


        # 3D 볼륨 로드 및 전처리
        img_itk = sitk.ReadImage(src_path)
        img_np  = sitk.GetArrayFromImage(img_itk)        # shape = (Z, H, W)
        max98   = np.percentile(img_np, 98)
        img_np  = np.clip(img_np, 0, max98) / max98     # 0~1

        img_np  = img_np * 2.0 - 1.0                    # -1~1

        # print("flip사용 주의!!!!!!!!!!!!!")
        img_np = np.flip(img_np, axis=-2).copy()
        img_np = img_np[f, :]

        # 빈 슬라이스 제거 & 홀수 슬라이스만 선택
        filt_idx = [i for i in range(len(img_np)) if img_np[i].max() != 0]
        filt_vol = img_np[filt_idx]

        # if len(filt_vol_) % 2 == 0:
        #     continue
        # odd_idx  = list(range(0, len(filt_vol_), 2))
        # filt_vol = filt_vol_[odd_idx]                    # shape = (D', H, W)

        # print("odd_idx:", odd_idx)


        tensor_img = torch.from_numpy(filt_vol).float()
        input_tensor=tensor_img.unsqueeze(1).to(device)
        # latents_tensor=latents_tensor.to(device)
        cond_tensor = model.module.encode(input_tensor)
        xTs         = model.module.encode_stochastic(input_tensor, cond_tensor, T=t)


    # ─── 4) interpolation & 저장 루프 ─────────────────────────────────────────
    for src_path in tqdm(files, desc="Generating"):
        img_id = os.path.basename(src_path).split(".")[0]
        id = img_id.split("_")[2]

        save_to_root = os.path.join("/storage/kjh/dataset/cardiac/output/DiffAE", f"{id}")
        os.makedirs(save_to_root, exist_ok=True)

        # 4.1) 3D 볼륨 로드 및 전처리
        img_itk = sitk.ReadImage(src_path)
        vol_np  = sitk.GetArrayFromImage(img_itk)           # (Z, H, W)
        max98   = np.percentile(vol_np, 98)
        vol_np  = np.clip(vol_np, 0, max98) / max98         # [0,1]
        vol_np  = vol_np * 2.0 - 1.0                        # [-1,1]
        vol_np  = np.flip(vol_np, axis=-2).copy()           # depth 방향 flip

        # 4.2) 관심 frame(f) 추출
        frame_vol = vol_np[f]  # shape: (D, H, W)

        # 4.3) 빈 슬라이스 제거
        nonzero_indices = [i for i in range(frame_vol.shape[0]) if frame_vol[i].max() != 0]
        filt_vol        = frame_vol[nonzero_indices]  # (D', H, W)

        # 4.5) Tensor 변환 & GPU 전송
        tensor_vol = torch.from_numpy(filt_vol).unsqueeze(1).float().to(device)  # [6,1,H,W]
        # encode / encode_stochastic 은 반드시 model.module.xxx 으로 호출
        cond_tensor = model.module.encode(tensor_vol)  
        xT_tensor   = model.module.encode_stochastic(tensor_vol, cond_tensor, T=t)

        # 4.6) 보간 작업: numpy로 내려서 처리
        xT_np   = xT_tensor.cpu().numpy()    # [6, latent_dim]
        cond_np = cond_tensor.cpu().numpy()  # [6, cond_dim]

        final_xT_list   = []
        final_cond_list = []
        for i in range(xT_np.shape[0] - 1):
            final_xT_list  .append(xT_np[i])
            final_cond_list.append(cond_np[i])
            final_xT_list  .extend(slerp_interpolate(xT_np[i],   xT_np[i+1], num=filling))
            final_cond_list.extend(lin_interpolate   (cond_np[i], cond_np[i+1], num=filling))
        final_xT_list  .append(xT_np[-1])
        final_cond_list.append(cond_np[-1])

        interp_xT   = torch.from_numpy(np.stack(final_xT_list)).float().to(device)
        interp_cond = torch.from_numpy(np.stack(final_cond_list)).float().to(device)

        # 4.7) 보간된 전체 시퀀스로 렌더링 (multi-GPU)
        with torch.no_grad():
            # LitModel.forward 에서 render를 호출하도록 정의되어 있으면 아래 한 줄로 동작
            pred = model.module.render(interp_xT, cond=interp_cond, T=t)
            # 만약 LitModel.forward가 render를 래핑하지 않았다면:
            # pred = model.module.render(interp_xT, cond=interp_cond, T=t)

        pred_np = pred.squeeze(1).cpu().numpy()  # (Z_interp, H, W)

        # 4.8) 후처리: intensity 복원 및 flip 복구
        pred_np = pred_np * max98
        pred_np = np.flip(pred_np, axis=1).copy()  # height 축 flip 복구

        # 4.9) SimpleITK 이미지 변환 및 메타데이터 설정
        out_itk = sitk.GetImageFromArray(pred_np)
        orig_sp = img_itk.GetSpacing()             # (x, y, z)
        orig_n  = len(filt_vol)                    # 입력 slice 개수 (6)
        new_n   = (orig_n - 1) * (filling - 1) + orig_n
        new_z   = orig_sp[2] * (orig_n-1) / (new_n - 1)
        out_itk.SetSpacing((orig_sp[0], orig_sp[1], new_z))
        out_itk.SetOrigin(img_itk.GetOrigin())
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

        # 4.10) 출력 경로에 쓰기
        save_path = os.path.join(save_to_root, f"{img_id}_{f}.nii.gz")
        sitk.WriteImage(out_itk, save_path)