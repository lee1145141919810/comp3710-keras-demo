"""Convolutional Variational Autoencoder (Kingma & Welling, 2014) for OASIS brain slices.

Encoder   x [1, S, S]  ->  conv stack (stride 2, S -> 8x8)  ->  mu, log_var  in R^latent_dim
Sampling  z = mu + sigma * eps,  eps ~ N(0, I)                      (re-parameterisation trick)
Decoder   z  ->  linear  ->  transposed-conv stack (8x8 -> S)  ->  sigmoid  ->  x_hat in [0, 1]

The number of down/up-sampling stages is derived from the image size, so the same class works for
64x64, 128x128 and the native 256x256 slices.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """4x4 stride-2 convolution halving the resolution, with BatchNorm + LeakyReLU."""
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True))


def deconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """4x4 stride-2 transposed convolution doubling the resolution, with BatchNorm + ReLU."""
    return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class ConvVAE(nn.Module):
    def __init__(self, image_size: int = 128, latent_dim: int = 2, base_channels: int = 32, bottleneck: int = 8):
        super().__init__()
        assert image_size % bottleneck == 0 and (image_size // bottleneck) & (image_size // bottleneck - 1) == 0, "image_size / 8 must be a power of two"
        n_stages = int(math.log2(image_size // bottleneck))
        channels = [min(base_channels * 2**i, 512) for i in range(n_stages)]  # 32, 64, 128, 256, (512)
        self.latent_dim, self.bottleneck, self.top_channels = latent_dim, bottleneck, channels[-1]

        enc, in_ch = [], 1
        for ch in channels:
            enc.append(conv_block(in_ch, ch))
            in_ch = ch
        self.encoder = nn.Sequential(*enc, nn.Flatten())
        flat = self.top_channels * bottleneck * bottleneck
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flat)
        dec = []
        for in_ch, out_ch in zip(channels[::-1], channels[::-1][1:]):
            dec.append(deconv_block(in_ch, out_ch))
        dec.append(nn.ConvTranspose2d(channels[0], 1, 4, stride=2, padding=1))  # final layer: no BN, sigmoid later
        self.decoder = nn.Sequential(*dec)

    # -- pieces -------------------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, sigma^2) in a differentiable way: z = mu + sigma * eps."""
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, self.top_channels, self.bottleneck, self.bottleneck)
        return torch.sigmoid(self.decoder(h))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        return self.decode(z), mu, log_var


def vae_loss(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, beta: float = 1.0, recon: str = "bce") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Negative ELBO per image = reconstruction term + beta * KL(q(z|x) || N(0, I)).

    Both terms are summed over pixels / latent dimensions and averaged over the batch.
    Returns (total, reconstruction, kl).
    """
    if recon == "bce":
        rec = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.shape[0]
    else:
        rec = F.mse_loss(x_hat, x, reduction="sum") / x.shape[0]
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.shape[0]
    return rec + beta * kl, rec, kl


if __name__ == "__main__":
    for size in (64, 128, 256):
        m = ConvVAE(image_size=size, latent_dim=2)
        x_hat, mu, lv = m(torch.rand(2, 1, size, size))
        print(f"image {size}: recon {tuple(x_hat.shape)}, mu {tuple(mu.shape)}, params {sum(p.numel() for p in m.parameters()):,}")
