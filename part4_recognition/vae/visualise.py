"""Visualisations of a trained VAE: reconstructions, prior samples, the 2-D latent manifold and a
UMAP embedding of the latent codes. Used at the end of ``train.py`` and runnable on its own:

    python part4_recognition/vae/visualise.py --checkpoint results/part4_vae/vae_latent2.pt --data_root ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `common` importable

import matplotlib.pyplot as plt  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import RESULTS_DIR, get_device, save_figure  # noqa: E402
from part4_recognition.oasis_data import OASISDataset  # noqa: E402
from part4_recognition.vae.model import ConvVAE  # noqa: E402

OUT_DIR = RESULTS_DIR / "part4_vae"


@torch.no_grad()
def encode_dataset(model: ConvVAE, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return the posterior means ``mu`` [N, latent_dim] and slice indices [N] for a whole loader."""
    model.eval()
    mus, slices = [], []
    for x, s in loader:
        mu, _ = model.encode(x.to(device))
        mus.append(mu.cpu())
        slices.append(torch.as_tensor(s))
    return torch.cat(mus).numpy(), torch.cat(slices).numpy()


@torch.no_grad()
def plot_reconstructions(model: ConvVAE, x: torch.Tensor, device: torch.device, path: Path, n: int = 10) -> None:
    model.eval()
    x = x[:n].to(device)
    x_hat, _, _ = model(x)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.4))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(x_hat[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_title("input", loc="left", fontsize=9)
    axes[1, 0].set_title("reconstruction", loc="left", fontsize=9)
    fig.tight_layout()
    save_figure(fig, path)


@torch.no_grad()
def plot_prior_samples(model: ConvVAE, device: torch.device, path: Path, n: int = 32) -> None:
    """Decode z ~ N(0, I): shows what the generative model alone produces."""
    model.eval()
    z = torch.randn(n, model.latent_dim, device=device)
    imgs = model.decode(z).cpu()[:, 0]
    cols = 8
    fig, axes = plt.subplots(n // cols, cols, figsize=(1.6 * cols, 1.7 * n // cols))
    for ax, img in zip(axes.ravel(), imgs):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle("samples decoded from z ~ N(0, I)")
    fig.tight_layout()
    save_figure(fig, path)


@torch.no_grad()
def plot_manifold_2d(model: ConvVAE, device: torch.device, path: Path, n: int = 15, span: float = 0.97) -> None:
    """Classic VAE manifold: decode a grid of z whose coordinates are Gaussian quantiles.

    Using the inverse CDF of N(0, 1) (rather than a uniform grid) spreads the grid according to the
    prior, so every cell is roughly equally likely under p(z).
    """
    assert model.latent_dim == 2, "manifold grid needs a 2-D latent space"
    model.eval()
    quantiles = norm.ppf(np.linspace(1 - span, span, n))
    zx, zy = np.meshgrid(quantiles, quantiles[::-1])  # y decreasing so the plot reads like a map
    z = torch.as_tensor(np.stack([zx.ravel(), zy.ravel()], 1), dtype=torch.float32, device=device)
    imgs = torch.cat([model.decode(zb) for zb in z.split(64)]).cpu()[:, 0].numpy()
    s = imgs.shape[-1]
    canvas = imgs.reshape(n, n, s, s).transpose(0, 2, 1, 3).reshape(n * s, n * s)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, extent=[quantiles[0], quantiles[-1], quantiles[0], quantiles[-1]])
    ax.set_xlabel("z_1")
    ax.set_ylabel("z_2")
    ax.set_title(f"VAE latent manifold ({n}x{n} grid over N(0,1) quantiles)")
    save_figure(fig, path)


def plot_latent_scatter(mu: np.ndarray, slices: np.ndarray, path: Path, title: str, xlabel: str = "z_1", ylabel: str = "z_2") -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(mu[:, 0], mu[:, 1], c=slices, cmap="viridis", s=4, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="slice index within the volume")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    save_figure(fig, path)


def plot_umap(mu: np.ndarray, slices: np.ndarray, path: Path) -> bool:
    """UMAP projection of the latent means (any latent_dim). Returns False if umap is not installed."""
    try:
        import umap  # noqa: F401 (optional dependency)
    except ImportError:
        print("umap-learn is not installed - skipping the UMAP plot (pip install umap-learn)")
        return False
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(mu)
    plot_latent_scatter(emb, slices, path, f"UMAP of VAE latent means (latent_dim={mu.shape[1]})", "UMAP-1", "UMAP-2")
    return True


@torch.no_grad()
def plot_pca_manifold(model: ConvVAE, mu: np.ndarray, device: torch.device, path: Path, n: int = 12) -> None:
    """For latent_dim > 2: decode a grid spanning the two main PCA directions of the encoded data."""
    model.eval()
    centre = mu.mean(0)
    _, S, Vt = np.linalg.svd(mu - centre, full_matrices=False)
    std = S[:2] / np.sqrt(len(mu) - 1)
    steps = np.linspace(-2.0, 2.0, n)
    grid = np.array([centre + a * std[0] * Vt[0] + b * std[1] * Vt[1] for b in steps[::-1] for a in steps], dtype=np.float32)
    imgs = torch.cat([model.decode(zb) for zb in torch.as_tensor(grid, device=device).split(64)]).cpu()[:, 0].numpy()
    s = imgs.shape[-1]
    canvas = imgs.reshape(n, n, s, s).transpose(0, 2, 1, 3).reshape(n * s, n * s)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, extent=[-2, 2, -2, 2])
    ax.set_xlabel("1st principal direction of the latent codes (std units)")
    ax.set_ylabel("2nd principal direction (std units)")
    ax.set_title(f"Decoded 2-D slice through the {mu.shape[1]}-D latent space")
    save_figure(fig, path)


@torch.no_grad()
def plot_interpolations(model: ConvVAE, x: torch.Tensor, device: torch.device, path: Path, n_pairs: int = 4, steps: int = 8) -> None:
    """Linear interpolation between the codes of pairs of real slices - a smooth manifold gives smooth morphs."""
    model.eval()
    x = x[: 2 * n_pairs].to(device)
    mu, _ = model.encode(x)
    fig, axes = plt.subplots(n_pairs, steps, figsize=(1.5 * steps, 1.6 * n_pairs))
    for r in range(n_pairs):
        a, b = mu[2 * r], mu[2 * r + 1]
        z = torch.stack([a + (b - a) * t for t in torch.linspace(0, 1, steps, device=device)])
        imgs = model.decode(z).cpu()[:, 0]
        for c in range(steps):
            axes[r, c].imshow(imgs[c], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
    fig.suptitle("latent-space interpolation between pairs of test slices")
    fig.tight_layout()
    save_figure(fig, path)


def make_all_figures(model: ConvVAE, test_ds: OASISDataset, device: torch.device, out_dir: Path, tag: str) -> None:
    """Produce every figure that applies to the model's latent dimensionality."""
    loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    x_batch = torch.stack([test_ds[i][0] for i in range(0, len(test_ds), max(1, len(test_ds) // 16))][:16])
    plot_reconstructions(model, x_batch, device, out_dir / f"reconstructions_{tag}.png")
    plot_prior_samples(model, device, out_dir / f"prior_samples_{tag}.png")
    plot_interpolations(model, x_batch, device, out_dir / f"interpolations_{tag}.png")

    mu, slices = encode_dataset(model, loader, device)
    if model.latent_dim == 2:
        plot_manifold_2d(model, device, out_dir / f"manifold_{tag}.png")
        plot_latent_scatter(mu, slices, out_dir / f"latent_scatter_{tag}.png", "Encoded test slices in the 2-D latent space")
    else:
        plot_pca_manifold(model, mu, device, out_dir / f"manifold_pca_{tag}.png")
    plot_umap(mu, slices, out_dir / f"umap_{tag}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split", default="test", choices=["train", "validate", "test"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ConvVAE(image_size=ckpt["image_size"], latent_dim=ckpt["latent_dim"], base_channels=ckpt["base_channels"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    ds = OASISDataset(args.data_root, args.split, image_size=ckpt["image_size"])
    make_all_figures(model, ds, device, OUT_DIR, tag=f"latent{ckpt['latent_dim']}")


if __name__ == "__main__":
    main()
