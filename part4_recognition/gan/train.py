"""Part 4 Task 3 - generate realistic brain slices with a DCGAN trained on OASIS.

    python part4_recognition/gan/train.py --data_root /path/to/keras_png_slices_data --image_size 128 --epochs 60

Evidence of training saved to results/part4_gan/:
    samples/epoch_XXX.png   fixed-noise sample grid after every epoch (watch the brains emerge)
    losses.png              generator / discriminator losses and D(x), D(G(z)) per epoch
    final_samples.png       64 samples from the final generator
    evolution.png           the same latent vectors decoded at several epochs
    diversity.json          pairwise-distance statistics (real vs generated) as a mode-collapse check
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

import matplotlib.pyplot as plt  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.utils import make_grid, save_image  # noqa: E402

from common import RESULTS_DIR, ensure_dir, get_device, save_figure, set_seed  # noqa: E402
from common.device import describe_device  # noqa: E402
from part4_recognition.gan.model import Discriminator, Generator, weights_init  # noqa: E402
from part4_recognition.oasis_data import OASISDataset  # noqa: E402

OUT_DIR = RESULTS_DIR / "part4_gan"


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] (tanh range) -> [0, 1] for saving / plotting."""
    return (x.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def diversity_stats(fake: torch.Tensor, real: torch.Tensor) -> dict[str, float]:
    """Mean pairwise L2 distance within a batch of generated vs real images.

    A collapsed generator produces near-identical images -> tiny within-fake distances. Healthy
    training gives a fake/real ratio close to 1. Also reports the mean distance of each fake image to
    its nearest real neighbour (large -> the generator is not simply memorising training slices).
    """
    f, r = fake.flatten(1).float().cpu(), real.flatten(1).float().cpu()  # tiny: 64 x pixels
    d_ff, d_rr = torch.cdist(f, f), torch.cdist(r, r)
    off = ~torch.eye(len(f), dtype=torch.bool)
    off_r = ~torch.eye(len(r), dtype=torch.bool)
    nn_dist = torch.cdist(f, r).min(dim=1).values
    return {
        "mean_pairwise_dist_fake": d_ff[off].mean().item(),
        "mean_pairwise_dist_real": d_rr[off_r].mean().item(),
        "fake_over_real_ratio": (d_ff[off].mean() / d_rr[off_r].mean()).item(),
        "mean_nearest_real_dist": nn_dist.mean().item(),
    }


def plot_losses(history: dict[str, list[float]], path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["loss_d"]) + 1)
    ax1.plot(epochs, history["loss_d"], label="discriminator")
    ax1.plot(epochs, history["loss_g"], label="generator")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE loss")
    ax1.set_title("GAN losses")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history["d_real"], label="D(x)  (real)")
    ax2.plot(epochs, history["d_fake"], label="D(G(z))  (fake)")
    ax2.axhline(0.5, color="k", linestyle="--", alpha=0.4)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("mean discriminator probability")
    ax2.set_title("discriminator confidence (0.5 = cannot tell)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, path)


def plot_evolution(snapshots: list[tuple[int, torch.Tensor]], path: Path, n_show: int = 8) -> None:
    """Rows = epochs, columns = the same fixed latent vectors: shows how the samples sharpen over time."""
    fig, axes = plt.subplots(len(snapshots), n_show, figsize=(1.5 * n_show, 1.6 * len(snapshots)))
    axes = np.atleast_2d(axes)
    for r, (epoch, imgs) in enumerate(snapshots):
        for c in range(n_show):
            axes[r, c].imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        axes[r, 0].set_title(f"epoch {epoch}", loc="left", fontsize=8)
    fig.tight_layout()
    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=None, help="folder containing keras_png_slices_train/ ...")
    parser.add_argument("--image_size", type=int, default=128, help="64 trains much faster; 128 looks better")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1 (0.5 as in the DCGAN paper)")
    parser.add_argument("--real_label", type=float, default=0.9, help="one-sided label smoothing for real images")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers; the dataset is already in RAM so 0 is usually fastest")
    parser.add_argument("--snapshot_every", type=int, default=10, help="epochs between rows of evolution.png")
    parser.add_argument("--max_samples", type=int, default=None, help="limit training slices (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    sample_dir = ensure_dir(OUT_DIR / "samples")
    device = get_device(args.device)
    print(f"device: {describe_device(device)}")

    # Unsupervised: only the images are used, no masks.
    train_ds = OASISDataset(args.data_root, "train", args.image_size, normalise="signed", max_samples=args.max_samples)
    loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    print(f"{len(train_ds)} training slices at {args.image_size}x{args.image_size}, {len(loader)} steps per epoch")

    G = Generator(args.latent_dim, args.image_size, args.base_channels).to(device)
    D = Discriminator(args.image_size, args.base_channels).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    print(f"G: {sum(p.numel() for p in G.parameters()):,} params   D: {sum(p.numel() for p in D.parameters()):,} params")
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999))

    fixed_z = torch.randn(64, args.latent_dim, device=device)  # same noise every epoch -> comparable grids
    history: dict[str, list[float]] = {"loss_d": [], "loss_g": [], "d_real": [], "d_fake": []}
    snapshots: list[tuple[int, torch.Tensor]] = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        sums = np.zeros(4)
        for real, _ in loader:
            real = real.to(device, non_blocking=True)
            b = real.shape[0]

            # -- discriminator: real -> real_label, fake -> 0 ------------------------------------
            z = torch.randn(b, args.latent_dim, device=device)
            fake = G(z)
            logits_real, logits_fake = D(real), D(fake.detach())
            loss_d = F.binary_cross_entropy_with_logits(logits_real, torch.full_like(logits_real, args.real_label)) + \
                F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # -- generator: non-saturating loss, wants D(G(z)) -> 1 ------------------------------
            logits_fake = D(fake)
            loss_g = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            sums += np.array([loss_d.item(), loss_g.item(), torch.sigmoid(logits_real).mean().item(), torch.sigmoid(logits_fake).mean().item()])

        means = sums / len(loader)
        for k, v in zip(history, means):
            history[k].append(float(v))
        print(f"epoch {epoch:3d}/{args.epochs}  loss_D {means[0]:.3f}  loss_G {means[1]:.3f}  D(x) {means[2]:.3f}  D(G(z)) {means[3]:.3f}  [{time.time() - start:6.0f}s]")

        G.eval()
        with torch.no_grad():
            samples = to_unit(G(fixed_z)).cpu()
        save_image(samples, sample_dir / f"epoch_{epoch:03d}.png", nrow=8)
        if epoch == 1 or epoch % args.snapshot_every == 0 or epoch == args.epochs:
            snapshots.append((epoch, samples))
        torch.save({"generator": G.state_dict(), "discriminator": D.state_dict(), "args": vars(args), "epoch": epoch}, OUT_DIR / "dcgan.pt")

    # -- evidence of training -----------------------------------------------------------------
    plot_losses(history, OUT_DIR / "losses.png")
    plot_evolution(snapshots, OUT_DIR / "evolution.png")
    G.eval()
    with torch.no_grad():
        fake = G(torch.randn(64, args.latent_dim, device=device))
        grid = make_grid(to_unit(fake).cpu(), nrow=8, padding=2)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid[0], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(f"DCGAN samples after {args.epochs} epochs ({args.image_size}x{args.image_size})")
    save_figure(fig, OUT_DIR / "final_samples.png")

    n_real = min(64, len(train_ds))
    real_batch = torch.stack([train_ds[i][0] for i in np.random.choice(len(train_ds), n_real, replace=False)]).to(device)
    stats = diversity_stats(fake, real_batch)
    print("diversity check (mode collapse => fake/real ratio << 1):", json.dumps(stats, indent=2))
    with open(OUT_DIR / "diversity.json", "w") as f:
        json.dump({"diversity": stats, "history": history, "args": vars(args)}, f, indent=2)
    print(f"training finished in {time.time() - start:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
