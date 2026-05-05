# Denoising Diffusion Probabilistic Model (DDPM)

A DDPM implementation from scratch using pure PyTorch, trained on FFHQ face dataset to generate and reconstruct high-resolution face images.

## Live Demo
🔗 [HuggingFace Space](https://huggingface.co/spaces/ImranAliNaeem/diffusion-models-high-res-image-generation)

---

## Overview

This project implements a Denoising Diffusion Probabilistic Model (DDPM) that learns to generate realistic face images by progressively removing noise from pure Gaussian noise through a learned reverse diffusion process.

The model is trained entirely from scratch using base PyTorch layers — no pretrained diffusion pipelines or HuggingFace Diffusers are used.

---

## Architecture

### U-Net Backbone

The backbone is a simplified U-Net with time-step conditioning at every layer.

**Encoder:**
- `in_conv`: Conv2d(3 → 64) at 128×128
- `down1`: DownBlock(64 → 128) at 64×64
- `down2`: DownBlock(128 → 256) at 32×32
- `down3`: DownBlock(256 → 512) at 16×16

**Bottleneck:**
- ResidualBlock(512 → 512)
- SelfAttention(512) at **16×16** resolution
- ResidualBlock(512 → 512)

**Decoder:**
- `up0`: UpBlock(512 → 256) at 32×32 + SelfAttention(256) at **32×32**
- `up1`: UpBlock(256 → 128) at 64×64
- `up2`: UpBlock(128 → 64) at 128×128
- `up3`: FinalBlock(64 → 64) — finest skip connection, no upsample
- `out_conv`: Conv2d(64 → 3)

**Key Design Decisions:**
- GroupNorm(8) instead of BatchNorm for stable training
- SiLU activation throughout
- Time embedding injected into every ResidualBlock
- Skip connections at all encoder levels including finest (128×128)
- FinalBlock used at finest level to avoid unnecessary upsampling

### Time Embedding
- Sinusoidal positional embeddings
- Passed through `Linear → GELU → Linear` MLP
- Computed once per forward pass, shared across all blocks

### Self-Attention
- Applied at 16×16 (bottleneck) and 32×32 (decoder) resolution
- Scaled dot-product attention with GroupNorm
- Residual connection

---

## Diffusion Process

### Forward Process (Noising)
Gradually adds Gaussian noise to images over T timesteps:

```
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
```

### Reverse Process (Denoising)
U-Net predicts noise at each timestep. DDPM posterior mean:

```
μ_θ = (sqrt(ᾱ_{t-1}) * β_t * x̂_0 + sqrt(α_t) * (1 - ᾱ_{t-1}) * x_t) / (1 - ᾱ_t)
```

### Noise Schedule
- Linear beta schedule
- beta_start = 1e-4, beta_end = 0.02
- T = 500 timesteps

---

## Training Details

| Setting | Value |
|---------|-------|
| Dataset | FFHQ Face Dataset |
| Images Used | 30,000 |
| Image Size | 128 × 128 |
| Timesteps (T) | 500 |
| Epochs | 60 |
| Batch Size | 16 |
| Optimizer | AdamW |
| Learning Rate | 2e-4 |
| LR Scheduler | CosineAnnealingLR |
| Weight Decay | 1e-4 |
| Gradient Clipping | 1.0 |
| Mixed Precision | torch.amp (float16) |
| Hardware | Dual T4 GPU (Kaggle) |
| Total Parameters | 26M |
| Loss Function | MSE (noise prediction) |

---

## Results

### Image Generation
5 unique realistic faces generated from pure random Gaussian noise using 500 reverse diffusion steps.

### Image Reconstruction
Target image is noised to t=T-1, then reversed through the full diffusion chain.

### Quantitative Evaluation

| Metric | Score |
|--------|-------|
| PSNR | 11.36 dB |
| SSIM | 0.2131 |

---

## Project Structure

```
├── ddpm_kaggle.ipynb        # Complete training notebook (Kaggle)
├── app.py                   # Gradio app for HuggingFace deployment
├── requirements.txt         # Dependencies
└── README.md
```

---

## Key Features

- Pure PyTorch implementation — no pretrained models used
- Self-attention at 16×16 and 32×32 resolution levels
- FinalBlock for finest-level skip connection without upsample artifacts
- Checkpoint save and resume — training survives kernel restarts
- Best model checkpoint saved separately during training
- Mixed precision training with GradScaler
- Cosine LR scheduling for smooth convergence
- DataParallel for dual GPU training
- Gradio app with denoising step gallery visualization

---

## How To Run

### Training on Kaggle
1. Upload `ddpm_kaggle.ipynb` to Kaggle
2. Enable GPU T4 x2 accelerator
3. Add FFHQ dataset: `greatgamedota/ffhq-face-data-set`
4. Run all cells top to bottom
5. Training resumes automatically from checkpoint if kernel restarts

### Gradio App (Local)
```bash
pip install -r requirements.txt
python app.py
```

### HuggingFace Space
Upload these three files to a Gradio Space:
- `app.py`
- `ddpm_best.pth`
- `requirements.txt`

---

## Requirements

```
torch
torchvision
gradio
numpy
scikit-image
```

---

## References

- [Denoising Diffusion Probabilistic Models — Ho et al. 2020](https://arxiv.org/abs/2006.11239)
- [FFHQ Dataset — Karras et al.](https://github.com/NVlabs/ffhq-dataset)
- [Annotated Diffusion — HuggingFace Blog](https://huggingface.co/blog/annotated-diffusion)
