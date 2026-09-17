"""Part 4 Task 1 - train a convolutional VAE on the OASIS brain MRI slices.

    python part4_recognition/vae/train.py --data_root /path/to/keras_png_slices_data --latent_dim 2
    python part4_recognition/vae/train.py --latent_dim 32 --epochs 30      # richer model, UMAP manifold

Outputs (results/part4_vae/): training curves, checkpoint, reconstructions, prior samples,
latent-space interpolations and the manifold (2-D grid for latent_dim=2, PCA slice + UMAP otherwise).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `common` importable

from torch.utils.data import DataLoader  # noqa: E402

from common import RESULTS_DIR, ensure_dir, get_device, set_seed  # noqa: E402
from common.device import describe_device  # noqa: E402
from common.plotting import plot_curves  # noqa: E402
from part4_recognition.oasis_data import OASISDataset  # noqa: E402
from part4_recognition.vae.model import ConvVAE, vae_loss  # noqa: E402
from part4_recognition.vae.visualise import make_all_figures  # noqa: E402

OUT_DIR = RESULTS_DIR / "part4_vae"


def run_epoch(model: ConvVAE, loader: DataLoader, device: torch.device, beta: float, recon: str, optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    """One pass over ``loader``; trains when an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    with torch.set_grad_enabled(training):
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            x_hat, mu, log_var = model(x)
            loss, rec, kl = vae_loss(x_hat, x, mu, log_var, beta, recon)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            for k, v in zip(sums, (loss, rec, kl)):
                sums[k] += v.item() * len(x)
    return {k: v / len(loader.dataset) for k, v in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=None, help="folder containing keras_png_slices_train/ ...")
    parser.add_argument("--image_size", type=int, default=128, help="slices are resized to this (64 / 128 / 256)")
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0, help="weight of the KL term (beta-VAE)")
    parser.add_argument("--recon", choices=["bce", "mse"], default="bce")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers; the dataset is already in RAM so 0 is usually fastest")
    parser.add_argument("--max_samples", type=int, default=None, help="limit slices per split (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(OUT_DIR)
    device = get_device(args.device)
    print(f"device: {describe_device(device)}")

    train_ds = OASISDataset(args.data_root, "train", args.image_size, max_samples=args.max_samples)
    val_ds = OASISDataset(args.data_root, "validate", args.image_size, max_samples=args.max_samples)
    test_ds = OASISDataset(args.data_root, "test", args.image_size, max_samples=args.max_samples)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    print(f"train {len(train_ds)} / validate {len(val_ds)} / test {len(test_ds)} slices at {args.image_size}x{args.image_size}")

    model = ConvVAE(args.image_size, args.latent_dim, args.base_channels).to(device)
    print(f"ConvVAE latent_dim={args.latent_dim}: {sum(p.numel() for p in model.parameters()):,} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_recon": [], "val_recon": [], "train_kl": [], "val_kl": []}
    tag = f"latent{args.latent_dim}"
    ckpt_path = OUT_DIR / f"vae_{tag}.pt"
    best_val = float("inf")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, args.beta, args.recon, optimizer)
        va = run_epoch(model, val_loader, device, args.beta, args.recon, None)
        scheduler.step()
        for k in ("loss", "recon", "kl"):
            history[f"train_{k}"].append(tr[k])
            history[f"val_{k}"].append(va[k])
        print(f"epoch {epoch:3d}/{args.epochs}  train -ELBO {tr['loss']:9.2f} (recon {tr['recon']:8.2f} kl {tr['kl']:6.2f}) | "
              f"val -ELBO {va['loss']:9.2f} (recon {va['recon']:8.2f} kl {va['kl']:6.2f})  [{time.time() - start:6.0f}s]")
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({"state_dict": model.state_dict(), "image_size": args.image_size, "latent_dim": args.latent_dim,
                        "base_channels": args.base_channels, "epoch": epoch, "val_loss": best_val}, ckpt_path)

    print(f"training finished in {time.time() - start:.0f}s; best validation -ELBO {best_val:.2f} -> {ckpt_path}")
    plot_curves({"train": history["train_loss"], "validation": history["val_loss"]}, OUT_DIR / f"loss_{tag}.png", "epoch", "-ELBO per image", "VAE loss")
    plot_curves({"train recon": history["train_recon"], "val recon": history["val_recon"], "train KL": history["train_kl"], "val KL": history["val_kl"]},
                OUT_DIR / f"loss_terms_{tag}.png", "epoch", "nats per image", "VAE reconstruction / KL terms")
    with open(OUT_DIR / f"history_{tag}.json", "w") as f:
        json.dump({"history": history, "args": vars(args), "best_val_loss": best_val}, f, indent=2)

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    make_all_figures(model, test_ds, device, OUT_DIR, tag)


if __name__ == "__main__":
    main()
