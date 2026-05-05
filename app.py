import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gradio as gr

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 128
TIME_EMB_DIM = 256
T            = 500
beta_start   = 1e-4
beta_end     = 0.02

# ── Noise Schedule ────────────────────────────────────────────────────────────
betas                        = torch.linspace(beta_start, beta_end, T).to(device)
alphas                       = 1.0 - betas
alpha_cumprod                = torch.cumprod(alphas, dim=0)
alpha_cumprod_prev           = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
sqrt_alpha_cumprod           = torch.sqrt(alpha_cumprod)
sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
posterior_variance           = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)

schedule = {
    "betas"                       : betas,
    "alphas"                       : alphas,
    "alpha_cumprod"                : alpha_cumprod,
    "alpha_cumprod_prev"           : alpha_cumprod_prev,
    "sqrt_alpha_cumprod"           : sqrt_alpha_cumprod,
    "sqrt_one_minus_alpha_cumprod" : sqrt_one_minus_alpha_cumprod,
    "posterior_variance"           : posterior_variance,
}

# ── Model Classes ─────────────────────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half      = self.dim // 2
        freqs     = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args      = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sinusoidal = SinusoidalPosEmb(dim)
        self.mlp        = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        return self.mlp(self.sinusoidal(t))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1     = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1     = nn.GroupNorm(8, out_ch)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2     = nn.GroupNorm(8, out_ch)
        self.act       = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.skip      = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.res        = ResidualBlock(in_ch, out_ch, time_emb_dim)
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        return self.downsample(x), x


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.res      = ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = x[:, :, :skip.shape[2], :skip.shape[3]]
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)


class FinalBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim):
        super().__init__()
        self.res = ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h          = self.norm(x)
        qkv        = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v    = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn       = torch.softmax((q.transpose(-1, -2) @ k) * C ** -0.5, dim=-1)
        out        = (attn @ v.transpose(-1, -2)).transpose(-1, -2)
        return x + self.proj(out.reshape(B, C, H, W))


class UNet(nn.Module):
    def __init__(self, time_emb_dim=TIME_EMB_DIM):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.in_conv        = nn.Conv2d(3, 64, 3, padding=1)
        self.down1          = DownBlock(64,  128, time_emb_dim)
        self.down2          = DownBlock(128, 256, time_emb_dim)
        self.down3          = DownBlock(256, 512, time_emb_dim)
        self.bot1           = ResidualBlock(512, 512, time_emb_dim)
        self.bot_attn       = SelfAttention(512)
        self.bot2           = ResidualBlock(512, 512, time_emb_dim)
        self.up0            = UpBlock(512, 512, 256, time_emb_dim)
        self.up0_attn       = SelfAttention(256)
        self.up1            = UpBlock(256, 256, 128, time_emb_dim)
        self.up2            = UpBlock(128, 128,  64, time_emb_dim)
        self.up3            = FinalBlock(64, 64,  64, time_emb_dim)
        self.out_conv       = nn.Conv2d(64, 3, 1)

    def forward(self, x, t):
        t_emb      = self.time_embedding(t)
        s1         = self.in_conv(x)
        s2, skip2  = self.down1(s1, t_emb)
        s3, skip3  = self.down2(s2, t_emb)
        bot, skip4 = self.down3(s3, t_emb)
        bot        = self.bot1(bot, t_emb)
        bot        = self.bot_attn(bot)
        bot        = self.bot2(bot, t_emb)
        x          = self.up0(bot, skip4, t_emb)
        x          = self.up0_attn(x)
        x          = self.up1(x, skip3, t_emb)
        x          = self.up2(x, skip2, t_emb)
        x          = self.up3(x, s1,    t_emb)
        return self.out_conv(x)


# ── Load Model ────────────────────────────────────────────────────────────────
model      = UNet().to(device)
checkpoint = torch.load("ddpm_best.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()


# ── Sampling Functions ────────────────────────────────────────────────────────
@torch.no_grad()
def p_sample(model, x_t, t_idx, schedule, device):
    t_tensor       = torch.tensor([t_idx], device=device).long()
    alpha_t        = schedule["alphas"][t_tensor]            .view(-1, 1, 1, 1)
    alpha_bar      = schedule["alpha_cumprod"][t_tensor]     .view(-1, 1, 1, 1)
    alpha_bar_prev = schedule["alpha_cumprod_prev"][t_tensor].view(-1, 1, 1, 1)
    beta_t         = schedule["betas"][t_tensor]             .view(-1, 1, 1, 1)
    post_var       = schedule["posterior_variance"][t_tensor].view(-1, 1, 1, 1)
    pred_noise     = model(x_t, t_tensor)
    x0_pred        = (x_t - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
    x0_pred        = x0_pred.clamp(-1, 1)
    mean = (
        torch.sqrt(alpha_bar_prev) * beta_t               * x0_pred
        + torch.sqrt(alpha_t)      * (1.0 - alpha_bar_prev) * x_t
    ) / (1.0 - alpha_bar)
    z      = torch.randn_like(x_t) if t_idx > 0 else torch.zeros_like(x_t)
    return mean + torch.sqrt(post_var.clamp(min=1e-20)) * z


@torch.no_grad()
def p_sample_loop(model, shape, schedule, device, T_steps):
    x          = torch.randn(shape, device=device)
    collect_at = set(torch.linspace(T_steps - 1, 0, 5).long().tolist())
    intermediates = []
    for t_idx in reversed(range(T_steps)):
        x = p_sample(model, x, t_idx, schedule, device)
        if t_idx in collect_at:
            img = (x[0].cpu().clamp(-1, 1) + 1) / 2
            intermediates.append(img)
    return x, intermediates


# ── Gradio Function ───────────────────────────────────────────────────────────
def generate_image():
    final, intermediates = p_sample_loop(
        model    = model,
        shape    = (1, 3, IMAGE_SIZE, IMAGE_SIZE),
        schedule = schedule,
        device   = device,
        T_steps  = T,
    )
    final_np = ((final[0].cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
    final_np = (final_np * 255).astype(np.uint8)
    steps_np = [
        (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        for img in intermediates
    ]
    return final_np, steps_np


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="DDPM Face Generator") as demo:
    gr.Markdown("# DDPM Face Generator")
    gr.Markdown("Generates a realistic face from pure random noise using a trained DDPM model.")
    btn        = gr.Button("Generate Image", variant="primary")
    output_img = gr.Image(label="Generated Face", type="numpy")
    gallery    = gr.Gallery(label="Denoising Steps (Noise → Face)", columns=5, height=200)
    btn.click(fn=generate_image, outputs=[output_img, gallery])

demo.launch()
