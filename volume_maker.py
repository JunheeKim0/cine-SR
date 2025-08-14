import os
import glob
import SimpleITK as sitk

# 1) 경로 설정
img_dir   = '/workspace/storage/kjh/dataset/cardiac/UK_bio/128_cine/train_400/img'
vol_dir   = '/workspace/storage/kjh/dataset/cardiac/UK_bio/128_cine/train_400/volume'
cfg_base  = '/workspace/storage/icml_data_collection/Cardiac_database/UKBiobank/UKBiobank_3956_300_600/train_'

os.makedirs(vol_dir, exist_ok=True)


# 2) 모든 cine 파일 순회
for img_path in glob.glob(os.path.join(img_dir, '*.nii.gz')):
    fname      = os.path.basename(img_path)
    # 예: "1010242_20209_2_0_crop_sa.nii.gz" → patient_id="1010242_20209_2_0"
    patient_id = fname.replace('_crop_sa.nii.gz', '')
    cfg_path   = os.path.join(cfg_base, patient_id, 'Info.cfg')

    # 3) Info.cfg 파싱: ED, ES, NbFrame 읽기
    ed = es = nbframe = None
    with open(cfg_path, 'r') as f:
        for line in f:
            key, val = line.strip().split(':')
            key = key.strip().lower()
            val = int(val.strip())
            if key == 'ed':
                ed = val - 1            # zero-based index 변환^[1]
            elif key == 'es':
                es = val - 1
            elif key == 'nbframe':
                nbframe = val

    # 4) 4D NIfTI 로드 → numpy array
    #    SimpleITK는 4D를 바로 지원하지 않을 수 있으므로, 
    #    (frames, H, W) 형태라면 3D cine이고, (frames, D, H, W)면 4D cine입니다.
    sitk_img = sitk.ReadImage(img_path)
    arr      = sitk.GetArrayFromImage(sitk_img)
    # arr.shape 예: (nbframe, depth, H, W) 또는 (nbframe, H, W)


    # 6) 각 frame → 3D volume으로 저장 (ED/ES 제외)
    for frame_idx in range(nbframe):
        if frame_idx in (ed, es):
            continue  # ED/ES 스킵

        # 3D 볼륨 뽑기
        vol3d = arr[frame_idx]   # shape = (D, H, W) 또는 (H, W)

        # 2D cine일 경우 (H, W)이면, depth=1인 3D로 변경
        if vol3d.ndim == 2:
            print("what..???????")

        # SimpleITK 이미지로 변환
        vol_img = sitk.GetImageFromArray(vol3d)

        # 원본 voxel spacing 복사 (첫 3차원만)
        orig_spacing = sitk_img.GetSpacing()  # e.g. (dx, dy, dz)
        vol_img.SetSpacing(orig_spacing)

        # 파일명: patientID_frameXX.nii.gz
        out_fname = f'{patient_id}_frame{frame_idx+1:02d}.nii.gz'
        out_path  = os.path.join(vol_dir, out_fname)

        sitk.WriteImage(vol_img, out_path)
