"""DCGAN (Radford et al., 2016) generator / discriminator for grayscale brain slices.

Generator     z ~ N(0, I) in R^latent_dim  ->  4x4 feature map  ->  transposed convs (x2 per stage)
              ->  tanh image in [-1, 1] of size ``image_size``.
Discriminator mirror image with strided convolutions and LeakyReLU; **spectral normalisation** on
              every layer (Miyato et al., 2018) keeps the discriminator Lipschitz-bounded, which is the
              single most effective stabiliser we found against oscillation / mode collapse.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128, image_size: int = 128, base_channels: int = 32):
        super().__init__()
        n_up = int(math.log2(image_size // 4))  # 4 -> image_size
        channels = [min(base_channels * 2**i, 512) for i in reversed(range(n_up))]  # e.g. 512,256,128,64,32
        self.latent_dim, self.start_channels = latent_dim, channels[0]
        self.project = nn.Sequential(nn.Linear(latent_dim, channels[0] * 4 * 4, bias=False), nn.BatchNorm1d(channels[0] * 4 * 4), nn.ReLU(inplace=True))

        layers: list[nn.Module] = []
        for in_ch, out_ch in zip(channels, channels[1:]):
            layers += [nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        layers += [nn.ConvTranspose2d(channels[-1], 1, 4, stride=2, padding=1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.project(z).view(-1, self.start_channels, 4, 4))


class Discriminator(nn.Module):
    def __init__(self, image_size: int = 128, base_channels: int = 32):
        super().__init__()
        n_down = int(math.log2(image_size // 4))
        channels = [min(base_channels * 2**i, 512) for i in range(n_down)]  # 32,64,128,256,512
        layers: list[nn.Module] = [spectral_norm(nn.Conv2d(1, channels[0], 4, stride=2, padding=1)), nn.LeakyReLU(0.2, inplace=True)]
        for in_ch, out_ch in zip(channels, channels[1:]):
            layers += [spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, inplace=True)]
        layers += [spectral_norm(nn.Conv2d(channels[-1], 1, 4, stride=1, padding=0))]  # 4x4 -> 1x1 logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1).squeeze(1)  # raw logits [B]


def weights_init(module: nn.Module) -> None:
    """DCGAN initialisation: N(0, 0.02) for conv weights, N(1, 0.02) for BatchNorm scales."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


if __name__ == "__main__":
    for size in (64, 128):
        g, d = Generator(image_size=size), Discriminator(image_size=size)
        fake = g(torch.randn(2, 128))
        print(f"{size}px: G {sum(p.numel() for p in g.parameters()):,} params -> {tuple(fake.shape)}; "
              f"D {sum(p.numel() for p in d.parameters()):,} params -> {tuple(d(fake).shape)}")
