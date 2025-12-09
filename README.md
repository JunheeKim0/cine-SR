### Hi there 👋🏻
<img width="2009" height="687" alt="Overall_architecture_white_background" src="https://github.com/user-attachments/assets/79d1e186-2bf8-43d2-bf77-61683998a858" />

````markdown
# Diffusion Autoencoders for Cardiac cine MRI

This repository is based on the **official implementation of Diffusion Autoencoders (DiffAE)**[^diffae-code] and adapted for **cardiac cine MRI** experiments.

We use DiffAE to learn a **strong generative prior** on cardiac short-axis (SAX) cine volumes and then solve a **3D cardiac inverse problem** via **long-axis (LAX)–conditioned inverse optimization**.

- **Base code**: Official DiffAE implementation (CVPR 2022, ORAL)[^diffae-paper]  
- **Data**: Only **cardiac cine MRI** (e.g., SAX volumes and LAX views); face/scene datasets like FFHQ/LSUN are **not used** in this fork  
- **Pipeline**:
  1. Train DiffAE (and optionally DDIM) on **cardiac cine MRI**.
  2. Run **inverse optimization** with `LAX_condition/ex8_multi_optimizer.py` to reconstruct anatomically plausible **3D cardiac volumes**.

---

## 1. Citation

If you use this codebase, please cite the original DiffAE paper and your corresponding cardiac MRI work.

```bibtex
@inproceedings{preechakul2021diffusion,
      title={Diffusion Autoencoders: Toward a Meaningful and Decodable Representation}, 
      author={Preechakul, Konpat and Chatthee, Nattanat and Wizadwongsa, Suttisak and Suwajanakorn, Supasorn},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
      year={2022},
}
````

## MGDCR Multi-View-Guided Diffusion Model for Cardiac Volume Reconstruction

This fork uses DiffAE as a **generative prior** for **cardiac cine MRI** and tackles **3D cardiac volume reconstruction** under sparse through-plane sampling.

* **Stage 1 – Prior learning (DiffAE)**
  Train a Diffusion Autoencoder on cardiac SAX cine MRI:

  * Semantic encoder → latent code (z_{\text{sem}}) (global anatomy).
  * Stochastic encoder → noisy latent (x_T) (fine detail).
  * Conditional DDIM decoder → predicts noise from ((x_t, t, z_{\text{sem}})).
    Training follows the standard **DDPM-style noise prediction objective**.

* **Stage 2 – Inverse optimization**
  Use the trained DiffAE as a prior and, given:

  * low through-plane **SAX** cine volume, and
  * multiple **LAX views** (2ch/3ch/4ch) with a unified alignment protocol,
    perform **LAX-conditioned inverse optimization** in latent space to reconstruct an anatomically plausible 3D cardiac volume that is consistent with the observations.

---

## 3. Installation

All dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

* A CUDA-enabled GPU is strongly recommended.
* Python / PyTorch versions can follow the original DiffAE implementation.
* For multi-GPU experiments, configure the environment as usual (`CUDA_VISIBLE_DEVICES`, etc.).

---

## 4. Data: Cardiac cine MRI only

> **Important**
> This fork **does not include any dataset**. You must prepare your own **cardiac cine MRI** data according to local ethics, IRB approval, and license constraints.

### 4.1 Expected data types

* **SAX cine MRI**

  * 2D slices over time, sparsely sampled through-plane (large slice spacing or missing slices).
* **LAX cine MRI**

  * Typical 2ch, 3ch, 4ch views.
  * Optionally pre-aligned by a unified protocol (e.g. standardized coverage and slice positions).

### 4.2 Pre-processing examples

The exact pre-processing is project-specific, but typical steps are:

* **Intensity normalization**

  * z-score, min–max, or mapping to ([-1, 1]).
* **Spatial pre-processing**

  * Center crop (e.g. (128 \times 128)).
  * Consistent orientation across SAX/LAX.
* **Splits**

  * Train / validation / test split at **subject level**.
* **Optional: LMDB format**

  * For efficient IO, you may convert the pre-processed slices/volumes into LMDB, following the original DiffAE style:

    ```text
    datasets/
    └─ cardiac_cine_128.lmdb
    ```

> In the original README, FFHQ/LSUN style datasets are used.
> In this fork, **all experiments are restricted to cardiac cine MRI**. Any mention of “FFHQ” in script names should be interpreted as “cardiac cine MRI”.

---

## 5. Training: Cardiac DiffAE prior

We reuse the original script names (e.g. `run_ffhq128.py`), but **the underlying dataset is replaced with cardiac cine MRI**.
The training code structure (configs, trainer, etc.) follows the original DiffAE implementation; only the dataset is changed.

### 5.1 DiffAE (cardiac, 128×128)

```bash
# Train DiffAE on cardiac cine MRI (SAX volumes)
python run_ffhq128.py
```

**Conceptual behavior**

* Trains a Diffusion Autoencoder on cardiac SAX cine MRI:

  * **Semantic encoder** ( z_{\mathrm{sem}} ): global anatomical structure (LV, RV, MYO, etc.).
  * **Stochastic encoder** ( x_T ): fine-grained, subject-specific variation.
  * **U-Net–based conditional DDIM decoder**: predicts noise from ((x_t, t, z_{\mathrm{sem}})).
* Uses the standard DiffAE objective:
  [
  \mathcal{L}*\text{DiffAE}
  = \mathbb{E}*{x_0, t, \epsilon}
  \left[
  \lVert
  \epsilon_\theta (x_t, t, \mathrm{Enc}_\phi(x_0)) - \epsilon
  \rVert_2^2
  \right].
  ]

**Outputs**

* Checkpoints, e.g.:

  ```text
  checkpoints/
  └─ cardiac_autoenc/
     ├─ last.ckpt         # DiffAE checkpoint
     └─ latent.ckpt       # (optional) precomputed z_sem for the dataset
  ```
* Training logs and sample images under `logs/` (depending on your logger/config).

> In the original project, this script trains on **FFHQ128**.
> In this fork, it is configured to train on **cardiac cine MRI** instead.

### 5.2 Optional: DDIM baseline (cardiac)

```bash
# Train DDIM baseline on cardiac cine MRI
python run_ffhq128_ddim.py
```

**Role**

* Trains a **pure DDIM/DDPM-style diffusion model** on cardiac cine MRI.
* Can be used as a **baseline generative prior** for comparison with DiffAE-based inverse optimization.
* Output checkpoints are stored similarly under `checkpoints/` (e.g., `checkpoints/cardiac_ddim/`).

> Other scripts like `run_ffhq256.py`, `run_bedroom128.py`, `run_horse128.py`, `run_celeba64.py` from the original DiffAE code are **not used** in this cardiac-specific fork.

---

## 6. Inverse Optimization: 3D cardiac volume reconstruction

After training DiffAE, we use the learned prior to reconstruct a **3D cardiac volume** consistent with SAX and LAX observations.

Inverse optimization is performed by:

```bash
python LAX_condition/ex8_multi_optimizer.py
```

### 6.1 Script location

* File: `LAX_condition/ex8_multi_optimizer.py`
* This script is specific to this fork and is **not part of the original DiffAE repository**.

### 6.2 Inputs (conceptual)

* **Trained DiffAE checkpoint**

  * e.g. `checkpoints/cardiac_autoenc/last.ckpt`.
* **Cardiac SAX cine MRI**

  * Low through-plane resolution stack (sparse slices / thick slices).
* **Cardiac LAX views**

  * 2ch, 3ch, 4ch volumes or slices, aligned to the same cardiac cycle and coordinate system as SAX.
* **Config / arguments**

  * The script may accept command line arguments (e.g. subject ID, frame index, number of optimization steps, etc.).
  * Please inspect `ex8_multi_optimizer.py` for project-specific options.

### 6.3 What the optimizer does

Conceptually, the script:

1. **Initializes** latent variables ((z_{\mathrm{sem}}, x_T)) or a 3D SAX volume estimate.
2. **Iteratively updates** the latent or volume estimate to:

   * Satisfy **SAX data fidelity** (match the observed SAX slices).
   * Satisfy **LAX consistency** (reprojected SAX volume matches LAX views).
   * Stay close to the **DiffAE prior** (anatomically plausible cardiac shapes).
3. Uses a **DiffAE-based denoising step** (diffusion prior) and an **optimization step** (gradient-based update) in alternation.

As a result, we obtain a reconstructed **3D high-resolution cardiac volume** that:

* Is consistent with the acquired SAX and LAX cine images.
* Respects the anatomical prior learned by DiffAE.

### 6.4 Outputs

* Reconstructed 3D cardiac volumes (e.g. NIfTI or other volumetric formats).
* Optional intermediate reconstructions (per optimization step) if enabled.
* Logs summarizing reconstruction time, loss curves, etc., depending on implementation.

---

## 7. Example end-to-end workflow

1. **Prepare cardiac dataset**

   * Preprocess SAX / LAX cine MRI.
   * Convert to the dataset format expected by `run_ffhq128.py` (e.g. LMDB).

2. **Train DiffAE**

   ```bash
   python run_ffhq128.py
   ```

   * Verify training curves and sample reconstructions from the checkpoint.

3. **(Optional) Train DDIM baseline**

   ```bash
   python run_ffhq128_ddim.py
   ```

4. **Run inverse optimization**

   ```bash
   python LAX_condition/ex8_multi_optimizer.py \
     --config <your_config> \
     --ckpt checkpoints/cardiac_autoenc/last.ckpt \
     --subject_id <subject_id> \
     --frame 0
   ```

   * The arguments above are **examples**; adjust them to match your config and the actual argument names in the script.

5. **Evaluate**

   * Compare reconstructed volumes to high-resolution references if available.
   * Compute segmentation-based metrics (e.g. Dice, HD95) and/or clinical indices (e.g. EDV, ESV, EF) as needed.

---

## 8. Checkpoints (summary)

Unlike the original DiffAE README, this fork does **not** distribute pretrained FFHQ/LSUN/etc. checkpoints.

* You are expected to **train DiffAE on your own cardiac cine MRI dataset**.

* Checkpoints should be stored in a directory like:

  ```text
  checkpoints/
  └─ cardiac_autoenc/
     ├─ last.ckpt
     └─ latent.ckpt    # optional, precomputed z_sem
  ```

* For other experiments (e.g., DDIM baseline), you may create additional directories under `checkpoints/`.

---

## 9. Repository structure (rough sketch)

A possible repository layout:

```text
.
├─ LAX_condition/
│  └─ ex8_multi_optimizer.py       # inverse optimization script (cardiac-specific)
├─ checkpoints/
│  └─ cardiac_autoenc/             # your trained DiffAE checkpoints
├─ datasets/
│  └─ cardiac_cine_128.lmdb        # your cardiac cine MRI dataset (optional LMDB)
├─ run_ffhq128.py                  # training script for DiffAE (cardiac)
├─ run_ffhq128_ddim.py             # training script for DDIM baseline (cardiac)
├─ requirements.txt
└─ README.md                       # this file
```

---

## 10. License and acknowledgements

This fork is derived from the official DiffAE implementation.[^diffae-code]

* Please check the **original DiffAE repository license** before using or redistributing this code.
* You are responsible for complying with all dataset licenses, IRB/ethics regulations, and local laws when using cardiac cine MRI data.

We gratefully acknowledge the authors of **Diffusion Autoencoders** for releasing their code and models.

---

[^diffae-paper]: Preechakul et al., *“Diffusion Autoencoders: Toward a Meaningful and Decodable Representation,”* CVPR 2022 (ORAL).

[^diffae-code]: Original DiffAE code: [https://github.com/phizaz/diffae](https://github.com/phizaz/diffae). This fork modifies their implementation to support cardiac cine MRI and an additional LAX-conditioned inverse optimization pipeline.

```
::contentReference[oaicite:0]{index=0}
```
