"""CIFAR-10 held entirely on the GPU with batched, on-device data augmentation.

The whole training set is only 50 000 x 3 x 32 x 32 values, so instead of a CPU ``DataLoader``
(which is the usual bottleneck in fast CIFAR training) the dataset is normalised once, reflect-padded
by 4 pixels and kept on the accelerator. Each epoch the standard augmentations are applied to the
*entire* tensor with a handful of vectorised ops:

    random crop 32x32 from the 40x40 padded image  ->  random horizontal flip  ->  8x8 cutout

This follows the approach of the DAWNBench "cifar10-fast" entries by David Page / Myrtle.ai.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
# The official Toronto URL 301s here; pinning the destination avoids a slow extra hop.
CIFAR10_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"


def _to_tensor(images_uint8: np.ndarray, pad: int = 0) -> torch.Tensor:
    """uint8 [N, 32, 32, 3] numpy -> normalised float32 [N, 3, 32+2p, 32+2p] tensor (CPU)."""
    x = torch.as_tensor(images_uint8).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    if pad:
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")  # pad once, crop every epoch
    return x


def load_cifar10(
    root: Path | str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    pad: int = 4,
    subset: int | None = None,
    download: bool = True,
) -> dict[str, torch.Tensor]:
    """Return {'train_x' (padded), 'train_y', 'test_x', 'test_y'} tensors living on ``device``.

    ``subset`` keeps only the first N training / test images (handy for CPU smoke tests).
    """
    torchvision.datasets.CIFAR10.url = CIFAR10_URL
    train = torchvision.datasets.CIFAR10(str(root), train=True, download=download)
    test = torchvision.datasets.CIFAR10(str(root), train=False, download=download)
    train_data, train_targets = train.data, np.asarray(train.targets)
    test_data, test_targets = test.data, np.asarray(test.targets)
    if subset:
        train_data, train_targets = train_data[:subset], train_targets[:subset]
        test_data, test_targets = test_data[:subset], test_targets[:subset]

    return {
        "train_x": _to_tensor(train_data, pad).to(device=device, dtype=dtype),
        "train_y": torch.as_tensor(train_targets, dtype=torch.long, device=device),
        "test_x": _to_tensor(test_data).to(device=device, dtype=dtype),
        "test_y": torch.as_tensor(test_targets, dtype=torch.long, device=device),
    }


@torch.no_grad()
def augment(padded: torch.Tensor, crop: int = 32, cutout: int = 8, flip: bool = True) -> torch.Tensor:
    """Random-crop + flip + cutout an entire padded dataset tensor [N, C, H+2p, W+2p] -> [N, C, crop, crop].

    Random crops are gathered offset-by-offset: for each of the (2p+1)^2 possible (dy, dx) offsets
    the images that drew that offset are sliced in one go, which is far cheaper than per-image ops.
    """
    n, c, hp, wp = padded.shape
    device = padded.device
    max_off = hp - crop
    dy = torch.randint(0, max_off + 1, (n,), device=device)
    dx = torch.randint(0, max_off + 1, (n,), device=device)

    out = torch.empty((n, c, crop, crop), device=device, dtype=padded.dtype)
    for y in range(max_off + 1):
        for x in range(max_off + 1):
            idx = torch.nonzero((dy == y) & (dx == x), as_tuple=True)[0]
            if idx.numel():
                out[idx] = padded[idx, :, y : y + crop, x : x + crop]

    if flip:
        mask = torch.rand(n, device=device) < 0.5
        out = torch.where(mask[:, None, None, None], out.flip(3), out)

    if cutout > 0:
        # Zero a random cutout x cutout square per image (centre may lie anywhere inside the image).
        cy = torch.randint(0, crop, (n,), device=device)
        cx = torch.randint(0, crop, (n,), device=device)
        ys = torch.arange(crop, device=device)[None, :, None]
        xs = torch.arange(crop, device=device)[None, None, :]
        y0, x0 = (cy - cutout // 2)[:, None, None], (cx - cutout // 2)[:, None, None]
        hole = (ys >= y0) & (ys < y0 + cutout) & (xs >= x0) & (xs < x0 + cutout)  # [N, crop, crop]
        out = out.masked_fill(hole[:, None], 0.0)
    return out
