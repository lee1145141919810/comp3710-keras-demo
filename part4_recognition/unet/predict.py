"""Inference / evaluation of a trained UNet on OASIS slices (used live during the demo).

    # evaluate the whole test set, print per-class DSC and save a figure of example segmentations
    python part4_recognition/unet/predict.py --checkpoint results/part4_unet/unet_best.pt --data_root ...

    # segment individual test slices
    python part4_recognition/unet/predict.py --checkpoint ... --indices 0 100 250
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `common` importable

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import RESULTS_DIR, ensure_dir, get_device, load_checkpoint, save_figure, synchronize  # noqa: E402
from common.device import describe_device  # noqa: E402
from part4_recognition.oasis_data import CLASS_NAMES, N_CLASSES, OASISDataset  # noqa: E402
from part4_recognition.unet.metrics import DiceAccumulator  # noqa: E402
from part4_recognition.unet.model import UNet  # noqa: E402

OUT_DIR = RESULTS_DIR / "part4_unet"
LABEL_CMAP = ListedColormap(["black", "#1f77b4", "#ff7f0e", "#fefefe"])  # background, CSF, grey, white


def load_model(checkpoint: str | Path, device: torch.device) -> tuple[UNet, dict]:
    ckpt = load_checkpoint(checkpoint, map_location=device)
    model = UNet(1, N_CLASSES, ckpt["base_channels"], ckpt["depth"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_labels(model: UNet, images: torch.Tensor) -> torch.Tensor:
    """Images ``[B, 1, H, W]`` -> predicted label maps ``[B, H, W]`` (argmax of the softmax output)."""
    probs = torch.softmax(model(images), dim=1)  # categorical / one-hot style output, one channel per class
    return probs.argmax(dim=1)


@torch.no_grad()
def evaluate_dataset(model: UNet, loader: DataLoader, device: torch.device) -> DiceAccumulator:
    """Run inference over ``loader`` and return the accumulated Dice statistics."""
    model.eval()
    acc = DiceAccumulator(N_CLASSES)
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        acc.update(predict_labels(model, images), masks)
    return acc


def format_dice_table(acc: DiceAccumulator, threshold: float = 0.9) -> str:
    per_class, per_image = acc.per_class(), acc.per_image_mean()
    lines = [f"{'class':<14}{'DSC (dataset)':>15}{'DSC (per-image mean)':>22}{'> ' + str(threshold):>8}"]
    for name, d, m in zip(CLASS_NAMES, per_class.tolist(), per_image.tolist()):
        lines.append(f"{name:<14}{d:>15.4f}{m:>22.4f}{'yes' if d > threshold else 'NO':>8}")
    lines.append(f"{'mean':<14}{per_class.mean():>15.4f}{per_image.mean():>22.4f}")
    lines.append(f"pixel accuracy: {acc.pixel_accuracy():.4f}")
    return "\n".join(lines)


@torch.no_grad()
def plot_predictions(model: UNet, ds: OASISDataset, indices: list[int], device: torch.device, path: Path) -> None:
    """Rows = slices; columns = MRI, ground truth, prediction, error map (red = wrong pixel)."""
    fig, axes = plt.subplots(len(indices), 4, figsize=(13, 3.3 * len(indices)))
    axes = np.atleast_2d(axes)
    for row, idx in enumerate(indices):
        image, mask = ds[idx]
        pred = predict_labels(model, image.unsqueeze(0).to(device))[0].cpu()
        per_class = DiceAccumulator(N_CLASSES)
        per_class.update(pred.unsqueeze(0), mask.unsqueeze(0))
        dsc = per_class.per_class().numpy()
        axes[row, 0].imshow(image[0], cmap="gray")
        axes[row, 0].set_title(f"{ds.image_files[idx].name}", fontsize=8)
        axes[row, 1].imshow(mask, cmap=LABEL_CMAP, vmin=0, vmax=N_CLASSES - 1, interpolation="nearest")
        axes[row, 1].set_title("ground truth", fontsize=9)
        axes[row, 2].imshow(pred, cmap=LABEL_CMAP, vmin=0, vmax=N_CLASSES - 1, interpolation="nearest")
        axes[row, 2].set_title("UNet prediction  DSC " + " / ".join(f"{d:.2f}" for d in dsc), fontsize=8)
        axes[row, 3].imshow(image[0], cmap="gray")
        axes[row, 3].imshow(np.ma.masked_where(pred == mask, np.ones_like(pred)), cmap=ListedColormap(["red"]), alpha=0.8, interpolation="nearest")
        axes[row, 3].set_title(f"errors ({(pred != mask).float().mean() * 100:.2f}% of pixels)", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("labels: black=background, blue=CSF, orange=grey matter, white=white matter", fontsize=9)
    fig.tight_layout()
    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(OUT_DIR / "unet_best.pt"))
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split", default="test", choices=["train", "validate", "test"])
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="slice indices to visualise (default: 4 spread over the split)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    ensure_dir(OUT_DIR)
    model, ckpt = load_model(args.checkpoint, device)
    print(f"device: {describe_device(device)}  checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch')}, val mean DSC {ckpt.get('val_dice', float('nan')):.4f})")

    ds = OASISDataset(args.data_root, args.split, ckpt["image_size"], with_masks=True, max_samples=args.max_samples)
    loader = DataLoader(ds, args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    synchronize(device)
    t0 = time.time()
    acc = evaluate_dataset(model, loader, device)
    synchronize(device)
    print(f"\ninference on {len(ds)} {args.split} slices took {time.time() - t0:.1f}s\n")
    print(format_dice_table(acc))

    indices = args.indices or list(np.linspace(0, len(ds) - 1, 4, dtype=int))
    plot_predictions(model, ds, indices, device, OUT_DIR / f"predictions_{args.split}.png")
    with open(OUT_DIR / f"dice_{args.split}.json", "w") as f:
        json.dump({"dsc_per_class": dict(zip(CLASS_NAMES, acc.per_class().tolist())),
                   "dsc_per_image_mean": dict(zip(CLASS_NAMES, acc.per_image_mean().tolist())),
                   "pixel_accuracy": acc.pixel_accuracy(), "n_slices": len(ds)}, f, indent=2)
    print(f"[saved] {OUT_DIR / f'dice_{args.split}.json'}")


if __name__ == "__main__":
    main()
