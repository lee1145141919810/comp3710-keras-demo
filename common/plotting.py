"""Matplotlib helpers. The Agg backend is forced so scripts also work on headless cluster nodes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after selecting the backend)
import numpy as np  # noqa: E402

from common.paths import ensure_dir  # noqa: E402


def save_figure(fig: plt.Figure, path: Path | str, dpi: int = 150) -> Path:
    """Save ``fig`` to ``path`` (creating the directory), close it and return the path."""
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")
    return path


def plot_image_grid(
    images: np.ndarray,
    path: Path | str,
    n_cols: int = 8,
    titles: list[str] | None = None,
    cmap: str = "gray",
    suptitle: str | None = None,
) -> Path:
    """Plot a batch of 2-D grayscale images ``[N, H, W]`` as a grid and save it."""
    images = np.asarray(images)
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.7 * n_rows))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i], cmap=cmap, vmin=images.min(), vmax=images.max())
            if titles is not None:
                ax.set_title(titles[i], fontsize=7)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_curves(curves: dict[str, list[float]], path: Path | str, xlabel: str, ylabel: str, title: str) -> Path:
    """Plot several named 1-D curves (e.g. train/val loss per epoch) on one axis."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in curves.items():
        ax.plot(range(1, len(values) + 1), values, marker="o", markersize=3, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return save_figure(fig, path)
