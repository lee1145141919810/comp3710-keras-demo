"""Part 4 Task 2 - train a UNet to segment OASIS brain slices into 4 tissue classes.

    python part4_recognition/unet/train.py --data_root /path/to/keras_png_slices_data --epochs 15 --amp

The network outputs one channel per class (softmax -> categorical / one-hot segmentation) and is
trained with cross-entropy + soft Dice loss. The validation split selects the best checkpoint by
mean DSC; the test split is evaluated once at the end (per-class DSC must exceed 0.9).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `common` importable

from torch.utils.data import DataLoader  # noqa: E402

from common import RESULTS_DIR, ensure_dir, get_device, set_seed  # noqa: E402
from common.device import describe_device  # noqa: E402
from common.plotting import plot_curves  # noqa: E402
from part4_recognition.oasis_data import CLASS_NAMES, N_CLASSES, OASISDataset, labels_to_one_hot  # noqa: E402
from part4_recognition.unet.metrics import DiceAccumulator, soft_dice_loss  # noqa: E402
from part4_recognition.unet.model import UNet  # noqa: E402
from part4_recognition.unet.predict import evaluate_dataset, format_dice_table, plot_predictions  # noqa: E402

OUT_DIR = RESULTS_DIR / "part4_unet"


def augment(images: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Light on-device augmentation: random left/right flip and random intensity scaling."""
    flip = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(flip[:, None, None, None], images.flip(3), images)
    masks = torch.where(flip[:, None, None], masks.flip(2), masks)
    gain = 1.0 + 0.1 * (2 * torch.rand(images.shape[0], 1, 1, 1, device=images.device) - 1)
    return (images * gain).clamp_(0, 1), masks


def train_one_epoch(model, loader, optimizer, scaler, device, amp_dtype, dice_weight: float, use_aug: bool) -> tuple[float, torch.Tensor]:
    model.train()
    loss_sum, acc = 0.0, DiceAccumulator(N_CLASSES)
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        if use_aug:
            images, masks = augment(images, masks)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
            loss = F.cross_entropy(logits, masks) + dice_weight * soft_dice_loss(logits.float(), labels_to_one_hot(masks, N_CLASSES))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(images)
        acc.update(logits.argmax(1), masks)
    return loss_sum / len(loader.dataset), acc.per_class()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=None, help="folder containing keras_png_slices_train/ ...")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=32, help="channels of the first UNet stage (32 -> 7.8M params)")
    parser.add_argument("--depth", type=int, default=4, help="number of pooling stages")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dice_weight", type=float, default=1.0, help="weight of the soft Dice term added to cross-entropy")
    parser.add_argument("--augment", action="store_true", help="random flips + intensity jitter")
    parser.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers; the dataset is already in RAM so 0 is usually fastest")
    parser.add_argument("--max_samples", type=int, default=None, help="limit slices per split (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(OUT_DIR)
    device = get_device(args.device)
    amp_dtype = None
    if args.amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.backends.cudnn.benchmark = True
    print(f"device: {describe_device(device)}  autocast: {amp_dtype}")

    train_ds = OASISDataset(args.data_root, "train", args.image_size, with_masks=True, max_samples=args.max_samples)
    val_ds = OASISDataset(args.data_root, "validate", args.image_size, with_masks=True, max_samples=args.max_samples)
    test_ds = OASISDataset(args.data_root, "test", args.image_size, with_masks=True, max_samples=args.max_samples)
    print(f"train {len(train_ds)} / validate {len(val_ds)} / test {len(test_ds)} slices; "
          f"train class fractions {dict(zip(CLASS_NAMES, np.round(train_ds.label_fractions(), 3).tolist()))}")
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    model = UNet(1, N_CLASSES, args.base_channels, args.depth).to(device)
    print(f"UNet(base={args.base_channels}, depth={args.depth}): {sum(p.numel() for p in model.parameters()):,} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    ckpt_path = OUT_DIR / "unet_best.pt"
    history: dict[str, list] = {"train_loss": [], "train_dice": [], "val_dice": []}
    best_val, start = -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, scaler, device, amp_dtype, args.dice_weight, args.augment)
        val_dice = evaluate_dataset(model, val_loader, device).per_class()
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice.tolist())
        history["val_dice"].append(val_dice.tolist())
        fmt = lambda t: "[" + " ".join(f"{v:.3f}" for v in t.tolist()) + "]"  # noqa: E731
        print(f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  train DSC {fmt(train_dice)}  "
              f"val DSC {fmt(val_dice)} (mean {val_dice.mean():.4f})  [{time.time() - t_epoch:.0f}s]")
        if val_dice.mean() > best_val:
            best_val = float(val_dice.mean())
            torch.save({"state_dict": model.state_dict(), "image_size": args.image_size, "base_channels": args.base_channels,
                        "depth": args.depth, "epoch": epoch, "val_dice": best_val}, ckpt_path)
    print(f"training finished in {time.time() - start:.0f}s; best validation mean DSC {best_val:.4f} -> {ckpt_path}")

    # -- curves -------------------------------------------------------------------------------
    plot_curves({"train": history["train_loss"]}, OUT_DIR / "loss.png", "epoch", "CE + Dice loss", "UNet training loss")
    val_curves = {name: [d[c] for d in history["val_dice"]] for c, name in enumerate(CLASS_NAMES)}
    plot_curves(val_curves, OUT_DIR / "val_dice.png", "epoch", "DSC", "Validation Dice per class")

    # -- final test evaluation with the best checkpoint ---------------------------------------
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    test_acc = evaluate_dataset(model, test_loader, device)
    print("\nTest set results (best checkpoint):")
    print(format_dice_table(test_acc))
    plot_predictions(model, test_ds, list(np.linspace(0, len(test_ds) - 1, 4, dtype=int)), device, OUT_DIR / "predictions_test.png")
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"test_dsc_per_class": dict(zip(CLASS_NAMES, test_acc.per_class().tolist())),
                   "test_pixel_accuracy": test_acc.pixel_accuracy(), "best_val_mean_dsc": best_val,
                   "history": history, "args": vars(args)}, f, indent=2)
    print(f"[saved] {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
