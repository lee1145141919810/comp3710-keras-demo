"""UNet (Ronneberger et al., 2015) for multi-class brain-tissue segmentation.

                 ┌──────────── skip ────────────┐
    x ─> DoubleConv(base) ──> pool ──> DoubleConv(2·base) ──> ... ──> bottleneck(16·base)
                 │                                                         │
    logits <─ 1x1 conv <─ DoubleConv(base) <─ concat/up <─ ... <──────────┘

* Contracting path: ``depth`` stages of (3x3 conv - BN - ReLU) x2 followed by 2x2 max-pooling; the
  channel count doubles each stage.
* Expanding path: 2x2 transposed convolution (x2 up-sampling), concatenation with the skip tensor of
  the same resolution, then another DoubleConv. The skip connections re-inject the fine spatial
  detail lost by pooling, which is what lets the network draw sharp tissue boundaries.
* Output: one channel per class -> ``softmax`` gives a categorical (one-hot style) segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    """(3x3 conv -> BatchNorm -> ReLU) x 2 keeping the spatial size."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class Up(nn.Module):
    """Transposed-conv up-sampling followed by skip concatenation and a DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)  # out_ch from the up-path + out_ch from the skip

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 4, base_channels: int = 32, depth: int = 4):
        super().__init__()
        chans = [base_channels * 2**i for i in range(depth + 1)]  # e.g. 32, 64, 128, 256, 512
        self.inc = DoubleConv(in_channels, chans[0])
        self.downs = nn.ModuleList(DoubleConv(chans[i], chans[i + 1]) for i in range(depth))
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList(Up(chans[i + 1], chans[i]) for i in reversed(range(depth)))
        self.outc = nn.Conv2d(chans[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(self.pool(skips[-1])))
        x = skips.pop()  # bottleneck
        for up in self.ups:
            x = up(x, skips.pop())
        return self.outc(x)  # logits [B, n_classes, H, W]; apply softmax for class probabilities


if __name__ == "__main__":
    net = UNet(base_channels=32)
    print(f"UNet parameters: {sum(p.numel() for p in net.parameters()):,}")
    print("logits:", net(torch.rand(2, 1, 256, 256)).shape)
