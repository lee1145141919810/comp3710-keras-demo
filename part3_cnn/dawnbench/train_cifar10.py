"""Part 3.2 of 4 - DAWNBench-style fast CIFAR-10 training with ResNet-18 (PyTorch).

Goal: >= 94 % test accuracy in a few minutes on a single GPU. Ingredients that make it fast:

    * ResNet-18 (CIFAR stem) trained from scratch - see ``resnet.py``.
    * Whole dataset resident on the GPU + vectorised on-device augmentation (``data.py``), so the
      input pipeline never stalls the GPU.
    * Mixed precision (``torch.autocast`` bf16 on Ampere/A100, fp16 + GradScaler on V100) and
      channels-last memory format to use the tensor cores.
    * SGD + Nesterov momentum, one-cycle *piecewise-linear* learning-rate schedule (warm-up to a
      high peak, then linear decay to 0), weight decay on conv/linear weights only, label smoothing.
    * Optional horizontal-flip test-time augmentation.

Typical runs (from the repository root):
    python part3_cnn/dawnbench/train_cifar10.py                          # full 30-epoch run
    python part3_cnn/dawnbench/train_cifar10.py --epochs 1               # one-epoch demo
    python part3_cnn/dawnbench/train_cifar10.py --eval_only --checkpoint results/part3_dawnbench/resnet18_cifar10.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `common` importable

from common import RESULTS_DIR, ensure_dir, get_device, set_seed, synchronize  # noqa: E402
from common.device import describe_device  # noqa: E402
from common.paths import DATA_DIR  # noqa: E402
from common.plotting import plot_curves  # noqa: E402
from part3_cnn.dawnbench.data import augment, load_cifar10  # noqa: E402
from part3_cnn.dawnbench.resnet import resnet18  # noqa: E402

OUT_DIR = RESULTS_DIR / "part3_dawnbench"


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------
def piecewise_linear_lr(step: int, total_steps: int, warmup_steps: int, peak: float) -> float:
    """0 -> ``peak`` linearly over the warm-up, then linearly back to 0 at ``total_steps``."""
    if step < warmup_steps:
        return peak * step / max(1, warmup_steps)
    return peak * max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def build_optimizer(model: nn.Module, lr: float, momentum: float, weight_decay: float) -> torch.optim.Optimizer:
    """SGD with Nesterov momentum; no weight decay on BatchNorm parameters and biases."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)


def autocast_dtype(device: torch.device, requested: str) -> torch.dtype | None:
    """Pick the mixed-precision dtype: bf16 where supported (A100), otherwise fp16; None disables AMP."""
    if requested == "none" or device.type == "cpu":
        return None
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if device.type != "cuda":  # "auto": autocast on MPS is still immature, keep fp32 there
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, amp_dtype: torch.dtype | None, tta: bool) -> tuple[float, float]:
    """Return (accuracy, mean loss) on ``x``/``y``. ``tta`` averages logits of the image and its mirror."""
    model.eval()
    correct, loss_sum = 0, 0.0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i : i + batch_size], y[i : i + batch_size]
        xb = xb.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=x.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(xb)
            if tta:
                logits = 0.5 * (logits + model(xb.flip(3)))
        loss_sum += F.cross_entropy(logits.float(), yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    return correct / len(x), loss_sum / len(x)


# --------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=str(DATA_DIR / "cifar10"), help="where CIFAR-10 is (or will be) downloaded")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--peak_lr", type=float, default=0.4, help="peak LR of the one-cycle schedule (for batch 512)")
    parser.add_argument("--warmup_epochs", type=float, default=5.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--cutout", type=int, default=8, help="side of the cutout square (0 disables)")
    parser.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto", help="mixed-precision mode")
    parser.add_argument("--tta", action="store_true", help="flip test-time augmentation for the final evaluation")
    parser.add_argument("--compile", action="store_true", help="wrap the model in torch.compile (PyTorch >= 2.0)")
    parser.add_argument("--target_acc", type=float, default=0.94, help="report the time at which this accuracy is first reached")
    parser.add_argument("--eval_only", action="store_true", help="skip training and only run inference on the test set")
    parser.add_argument("--checkpoint", default=None, help="state_dict to load before training / evaluation")
    parser.add_argument("--subset", type=int, default=None, help="use only the first N images (smoke tests)")
    parser.add_argument("--no_download", action="store_true", help="fail instead of downloading CIFAR-10 (offline compute nodes)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(OUT_DIR)
    device = get_device(args.device)
    amp_dtype = autocast_dtype(device, args.amp)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device: {describe_device(device)}   autocast dtype: {amp_dtype}")

    # -- data ---------------------------------------------------------------------------------
    t0 = time.time()
    # Keep the GPU-resident dataset in half precision only when autocast will consume it that way.
    data_dtype = torch.float16 if amp_dtype is not None else torch.float32
    data = load_cifar10(args.data_dir, device, dtype=data_dtype, subset=args.subset, download=not args.no_download)
    n_train = len(data["train_x"])
    print(f"data on device: train {tuple(data['train_x'].shape)}  test {tuple(data['test_x'].shape)}  ({time.time() - t0:.1f}s)")

    # -- model --------------------------------------------------------------------------------
    model = resnet18(num_classes=10).to(device).to(memory_format=torch.channels_last)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"loaded weights from {args.checkpoint}")
    print(f"ResNet-18 parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.compile:
        model = torch.compile(model)

    if args.eval_only:
        synchronize(device)
        t0 = time.time()
        acc, loss = evaluate(model, data["test_x"], data["test_y"], 1000, amp_dtype, args.tta)
        synchronize(device)
        print(f"inference on {len(data['test_x'])} test images: accuracy {acc:.4f}  loss {loss:.4f}  ({time.time() - t0:.2f}s)")
        return

    # -- training -----------------------------------------------------------------------------
    steps_per_epoch = (n_train + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    optimizer = build_optimizer(model, args.peak_lr, args.momentum, args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)  # only fp16 needs loss scaling

    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "test_acc": [], "test_loss": [], "epoch_time": []}
    step, train_time, reached_at = 0, 0.0, None
    print(f"training {args.epochs} epochs x {steps_per_epoch} steps, batch {args.batch_size}, peak lr {args.peak_lr}")
    print(f"{'epoch':>5} {'lr':>7} {'train loss':>10} {'train acc':>9} {'test loss':>9} {'test acc':>8} {'epoch s':>8} {'total s':>8}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        synchronize(device)
        t_epoch = time.time()

        x_aug = augment(data["train_x"], cutout=args.cutout)
        perm = torch.randperm(n_train, device=device)
        loss_sum, correct = torch.zeros((), device=device), torch.zeros((), device=device, dtype=torch.long)

        for i in range(0, n_train, args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb = x_aug[idx].contiguous(memory_format=torch.channels_last)
            yb = data["train_y"][idx]

            lr = piecewise_linear_lr(step, total_steps, warmup_steps, args.peak_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.detach() * len(idx)  # accumulate on device: no per-step host sync
            correct += (logits.argmax(1) == yb).sum()
            step += 1

        synchronize(device)
        epoch_time = time.time() - t_epoch
        train_time += epoch_time
        test_acc, test_loss = evaluate(model, data["test_x"], data["test_y"], 1000, amp_dtype, tta=False)
        train_loss, train_acc = loss_sum.item() / n_train, correct.item() / n_train
        for k, v in zip(("train_loss", "train_acc", "test_acc", "test_loss", "epoch_time"), (train_loss, train_acc, test_acc, test_loss, epoch_time)):
            history[k].append(v)
        if reached_at is None and test_acc >= args.target_acc:
            reached_at = (epoch, train_time)
        print(f"{epoch:>5d} {lr:>7.4f} {train_loss:>10.4f} {train_acc:>9.4f} {test_loss:>9.4f} {test_acc:>8.4f} {epoch_time:>8.1f} {train_time:>8.1f}")

    # -- final report -------------------------------------------------------------------------
    final_acc, _ = evaluate(model, data["test_x"], data["test_y"], 1000, amp_dtype, args.tta)
    print(f"\nfinal test accuracy: {final_acc:.4f}{' (with flip TTA)' if args.tta else ''}")
    print(f"total training time (excluding evaluation): {train_time:.1f}s")
    if reached_at:
        print(f"{args.target_acc:.0%} first reached at epoch {reached_at[0]} after {reached_at[1]:.1f}s of training")
    else:
        print(f"{args.target_acc:.0%} was not reached")

    ckpt_path = OUT_DIR / "resnet18_cifar10.pt"
    raw_model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    torch.save(raw_model.state_dict(), ckpt_path)
    plot_curves({"train": history["train_loss"], "test": history["test_loss"]}, OUT_DIR / "loss.png", "epoch", "cross-entropy", "ResNet-18 CIFAR-10 loss")
    plot_curves({"train": history["train_acc"], "test": history["test_acc"]}, OUT_DIR / "accuracy.png", "epoch", "accuracy", "ResNet-18 CIFAR-10 accuracy")
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(
            {
                "device": describe_device(device),
                "amp_dtype": str(amp_dtype),
                "final_test_accuracy": final_acc,
                "train_time_s": train_time,
                "target_reached": {"epoch": reached_at[0], "train_time_s": reached_at[1]} if reached_at else None,
                "history": history,
                "args": vars(args),
            },
            f,
            indent=2,
        )
    print(f"[saved] {ckpt_path}\n[saved] {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
