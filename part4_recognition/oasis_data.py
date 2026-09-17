"""Loader for the pre-processed OASIS brain MRI dataset (``keras_png_slices_data``).

Expected directory layout (the folder names are the ones shipped with the dataset; the root may
either contain them directly or contain a ``keras_png_slices_data`` sub-folder):

    <root>/
        keras_png_slices_train/          case_XXX_slice_Y.nii.png   (9 664 images, 256x256 gray)
        keras_png_slices_validate/       (1 120 images)
        keras_png_slices_test/           (544 images)
        keras_png_slices_seg_train/      seg_XXX_slice_Y.nii.png    (segmentation masks)
        keras_png_slices_seg_validate/
        keras_png_slices_seg_test/

Segmentation masks store the 4 tissue labels as the grey levels {0, 85, 170, 255}:
    0 background, 1 cerebrospinal fluid (CSF), 2 grey matter, 3 white matter.

Known locations of the data:
    * Rangpur cluster:  /home/groups/comp3710/OASIS
    * local copy:       set ``--data_root`` (e.g. ~/Downloads/keras_png_slices_data)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

N_CLASSES = 4
CLASS_NAMES = ("background", "CSF", "grey matter", "white matter")
LABEL_GREY_LEVELS = np.array([0, 85, 170, 255], dtype=np.float32)

# Candidate roots searched (in order) when --data_root is not given.
DEFAULT_ROOTS = (
    os.environ.get("OASIS_ROOT", ""),
    "/home/groups/comp3710/OASIS",
    "/home/groups/comp3710/OASIS/keras_png_slices_data",
    str(Path.home() / "Downloads" / "keras_png_slices_data"),
    str(Path(__file__).resolve().parents[1] / "data" / "keras_png_slices_data"),
)

_SPLIT_DIRS = {
    "train": ("keras_png_slices_train", "keras_png_slices_seg_train"),
    "validate": ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
    "test": ("keras_png_slices_test", "keras_png_slices_seg_test"),
}


def find_oasis_root(data_root: str | None = None) -> Path:
    """Resolve the directory that contains ``keras_png_slices_train`` etc.

    Accepts either the folder holding the split directories directly or its parent folder
    (``.../keras_png_slices_data``), and searches :data:`DEFAULT_ROOTS` when nothing is given.
    """
    candidates = [data_root] if data_root else list(DEFAULT_ROOTS)
    for cand in candidates:
        if not cand:
            continue
        for root in (Path(cand).expanduser(), Path(cand).expanduser() / "keras_png_slices_data"):
            if (root / "keras_png_slices_train").is_dir():
                return root
    raise FileNotFoundError(
        "Could not find the OASIS PNG slices. Pass --data_root pointing at the folder that contains "
        "'keras_png_slices_train' (or set the OASIS_ROOT environment variable). Tried: "
        + ", ".join(c for c in candidates if c)
    )


def parse_case_and_slice(filename: str) -> tuple[int, int]:
    """``case_123_slice_17.nii.png`` -> (123, 17). Returns (-1, -1) if the pattern does not match."""
    m = re.search(r"(\d+)_slice_(\d+)", filename)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def grey_levels_to_labels(mask: np.ndarray) -> np.ndarray:
    """Map mask grey levels {0, 85, 170, 255} -> class indices {0, 1, 2, 3} (nearest level wins)."""
    values = np.unique(mask)
    if np.isin(values, np.arange(N_CLASSES)).all():
        return mask.astype(np.int64)
    if not np.isin(values, LABEL_GREY_LEVELS).all():
        raise ValueError(f"Unknown mask grey levels: {values.tolist()}; expected 0/85/170/255 or 0/1/2/3")
    return (mask.astype(np.int64) // 85)


def labels_to_one_hot(labels: torch.Tensor, n_classes: int = N_CLASSES) -> torch.Tensor:
    """Integer label map ``[B, H, W]`` -> one-hot / categorical tensor ``[B, C, H, W]`` (float)."""
    return torch.nn.functional.one_hot(labels, n_classes).permute(0, 3, 1, 2).float()


class OASISDataset(Dataset):
    """In-memory dataset of OASIS PNG slices (+ optional segmentation masks).

    Every image is decoded once at construction and kept as ``uint8``, which keeps epoch time bounded
    by GPU compute rather than by PNG decoding on the shared cluster filesystem.

    Args:
        root: dataset root (see :func:`find_oasis_root`), or ``None`` to auto-detect.
        split: ``"train"``, ``"validate"`` or ``"test"``.
        image_size: images (bilinear) and masks (nearest) are resized to this square size.
        with_masks: also load the segmentation masks (needed for the UNet, not for VAE / GAN).
        normalise: ``"unit"`` -> [0, 1] (VAE / UNet), ``"signed"`` -> [-1, 1] (GAN with tanh output).
        max_samples: keep only the first N slices (quick experiments / smoke tests).

    ``__getitem__`` returns ``(image [1, H, W] float, mask [H, W] int64)`` when masks are loaded,
    otherwise ``(image, slice_index)`` so that VAE latent plots can be coloured by slice position.
    """

    def __init__(
        self,
        root: str | Path | None,
        split: str = "train",
        image_size: int = 256,
        with_masks: bool = False,
        normalise: str = "unit",
        max_samples: int | None = None,
    ):
        assert split in _SPLIT_DIRS, f"split must be one of {list(_SPLIT_DIRS)}"
        assert normalise in ("unit", "signed")
        self.root = find_oasis_root(str(root) if root else None)
        self.split, self.image_size, self.with_masks, self.normalise = split, image_size, with_masks, normalise

        img_dir, seg_dir = (self.root / d for d in _SPLIT_DIRS[split])
        self.image_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".png")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if max_samples:
            self.image_files = self.image_files[:max_samples]
        if not self.image_files:
            raise FileNotFoundError(f"no PNG files found in {img_dir}")

        self.cases = np.array([parse_case_and_slice(p.name)[0] for p in self.image_files])
        self.slices = np.array([parse_case_and_slice(p.name)[1] for p in self.image_files])

        self.images = self._load_stack(self.image_files, Image.BILINEAR, f"{split} images")
        self.masks: np.ndarray | None = None
        if with_masks:
            mask_files = self._resolve_masks(self.image_files, seg_dir)
            raw = self._load_stack(mask_files, Image.NEAREST, f"{split} masks")
            self.masks = grey_levels_to_labels(raw).astype(np.uint8)

    # -- loading helpers ----------------------------------------------------------------------
    @staticmethod
    def _matching_mask(image_path: Path, seg_dir: Path) -> Path:
        """Match a segmentation PNG to an image PNG.

        The official ``keras_png_slices_data`` archive uses ``case_XXX_slice_Y.nii.png`` next to
        ``seg_XXX_slice_Y.nii.png``. Some copies keep the same filename in both folders.
        """
        same = seg_dir / image_path.name
        if same.exists():
            return same
        swapped = seg_dir / image_path.name.replace("case_", "seg_", 1)
        if swapped.exists():
            return swapped
        key = parse_case_and_slice(image_path.name)
        matches = [p for p in seg_dir.glob("*.png") if key != (-1, -1) and parse_case_and_slice(p.name) == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous masks for {image_path.name}: {matches}")
        raise FileNotFoundError(f"no segmentation mask for {image_path.name} in {seg_dir}")

    @classmethod
    def _resolve_masks(cls, image_files: list[Path], seg_dir: Path) -> list[Path]:
        """Require identity-based pairing; sorted order cannot establish ground-truth identity."""
        masks = [cls._matching_mask(p, seg_dir) for p in image_files]
        if len(set(masks)) != len(masks):
            raise ValueError("Multiple images map to the same segmentation mask")
        return masks

    def _load_stack(self, files: list[Path], resample: int, desc: str) -> np.ndarray:
        out = np.empty((len(files), self.image_size, self.image_size), dtype=np.uint8)
        for i, path in enumerate(tqdm(files, desc=f"loading {desc}", unit="img", leave=False)):
            img = Image.open(path).convert("L")
            if img.size != (self.image_size, self.image_size):
                img = img.resize((self.image_size, self.image_size), resample)
            out[i] = np.asarray(img, dtype=np.uint8)
        return out

    # -- Dataset API --------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        image = torch.from_numpy(self.images[idx].astype(np.float32) / 255.0).unsqueeze(0)
        if self.normalise == "signed":
            image = image * 2.0 - 1.0
        if self.masks is not None:
            return image, torch.from_numpy(self.masks[idx].astype(np.int64))
        return image, int(self.slices[idx])

    def label_fractions(self) -> np.ndarray:
        """Fraction of pixels per class - shows how imbalanced the segmentation problem is."""
        assert self.masks is not None, "dataset was created without masks"
        counts = np.bincount(self.masks.ravel(), minlength=N_CLASSES)
        return counts / counts.sum()


if __name__ == "__main__":  # quick inspection:  python -m part4_recognition.oasis_data --data_root ...
    import argparse

    parser = argparse.ArgumentParser(description="Print basic statistics of the OASIS PNG dataset.")
    parser.add_argument("--data_root", default=None)
    args = parser.parse_args()
    for split in ("train", "validate", "test"):
        ds = OASISDataset(args.data_root, split, with_masks=True)
        img, mask = ds[0]
        print(f"{split:>8}: {len(ds):5d} slices from {len(np.unique(ds.cases))} cases | image {tuple(img.shape)} "
              f"range [{img.min():.2f}, {img.max():.2f}] | labels {torch.unique(mask).tolist()} | "
              f"class fractions {np.round(ds.label_fractions(), 3).tolist()}")
