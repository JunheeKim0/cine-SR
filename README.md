<img width="2009" height="500" alt="Overall_architecture_white_background" src="https://github.com/user-attachments/assets/79d1e186-2bf8-43d2-bf77-61683998a858" />

## MGDCR Multi-View-Guided Diffusion Model for Cardiac Volume Reconstruction
4D cine MRI is essential for cardiac function assessment. It offers continuous observation of ventricular dynamics, but its low through-plane resolution compromises anatomical continuity between slices and degrades the accuracy of essential clinical metrics, such as ventricular boundary delineation and ejection fraction (EF) calculation. We present LAG-DiffAE (Long-Axis Guided Diffusion Autoencoder), which integrates guidance from long-axis (LAX) views into an unsupervised through-plane interpolation framework. A diffusion autoencoder is pre-trained on 400 healthy-subject SAX volumes from UK Biobank and fine-tuned on 230 heart-failure cases. Given adjacent SAX slices, we synthesize intermediate planes by interpolating disentangled semantic and stochastic latents, then refine them via inverse optimization that aligns LAX-plane projections, enforces slice-to-slice continuity, and regularizes deviation from interpolated seeds. The method improves structural consistency and continuity over existing models, as shown by higher Dice and lower HD95, with visualizations confirming chamber-level coherence. This approach provides a practical solution to cine-MRI resolution limits and potential for future volumetric and functional analyses.
<!--This fork uses DiffAE as a **generative prior** for **cardiac cine MRI** and tackles **3D cardiac volume reconstruction** under sparse through-plane sampling.

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
    perform **LAX-conditioned inverse optimization** in latent space to reconstruct an anatomically plausible 3D cardiac volume that is consistent with the observations. -->

👉 [Project Page: MGDCR – Multi-View-Guided Diffusion Model for Cardiac Volume Reconstruction](https://junheekim0.github.io/Cardiac-Super-Resolution/ )
---

## Installation

```bash
pip install -r requirements.txt
```

---

## Data: Cardiac cine MRI only

### Expected data types

* **SAX cine MRI**
* **LAX cine MRI**
  * Typical 2ch, 3ch, 4ch views.

### Pre-processing examples

The exact pre-processing is project-specific, but typical steps are:

* **Intensity normalization**

  * z-score, min–max, or mapping to ([-1, 1]).
* **Spatial pre-processing**

  * Center crop (e.g. (128 x 128)).
  * Consistent orientation across SAX/LAX.

---

## Training: Diffusion autoencoder

```bash
# Train DiffAE on cardiac cine MRI (SAX volumes)
python run_medical_2nd.py
```

**Outputs**

* Checkpoints, e.g.:

  ```text
  checkpoints/
  └─ cardiac_autoenc/
     └─ last.ckpt  
     
  ```
* Training logs and sample images under `logs/` (depending on your logger/config).

---

## 6. Inverse Optimization: 3D cardiac volume reconstruction

After training DiffAE, we use the learned prior to reconstruct a **3D cardiac volume** consistent with SAX and LAX observations.

Inverse optimization is performed by:

```bash
python LAX_condition/ex8_lax_norm_final.py
```

### Inputs

* **Trained DiffAE checkpoint**

  * e.g. `checkpoints/cardiac_autoenc/last.ckpt`.
* **Cardiac SAX cine MRI**

  * Low through-plane resolution stack (sparse slices / thick slices).
* **Cardiac LAX views**

  * 2ch, 3ch, 4ch volumes or slices, aligned to the same cardiac cycle and coordinate system as SAX.
* **Config / arguments**

  * The script may accept command line arguments (e.g. subject ID, frame index, number of optimization steps, etc.).
  * Please inspect `ex8_lax_norm_final.py` for project-specific options.

### 6.4 Outputs

* Reconstructed 3D cardiac volumes (e.g. NIfTI or other volumetric formats).
* Optional intermediate reconstructions (per optimization step) if enabled.
* Logs summarizing reconstruction time, loss curves, etc., depending on implementation.

---

## 8. Checkpoints (summary)

Unlike the original DiffAE README, this fork does **not** distribute pretrained FFHQ/LSUN/etc. checkpoints.

* You are expected to **train DiffAE on your own cardiac cine MRI dataset**.

* Checkpoints should be stored in a directory like:

  ```text
  checkpoints/
  └─ cardiac_autoenc/
     └─ last.ckpt
      
  ```

---

## License and acknowledgements

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
