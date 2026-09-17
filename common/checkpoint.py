"""Checkpoint I/O that works across PyTorch versions.

PyTorch >= 2.6 defaults ``torch.load(..., weights_only=True)``, which refuses anything that is not a
tensor (ints, strings, nested dicts). Every checkpoint in this lab stores metadata next to the
``state_dict``, so we always request ``weights_only=False``.
"""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(obj: object, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    return path


def load_checkpoint(path: Path | str, map_location: torch.device | str | None = None) -> object:
    """Load a checkpoint written by :func:`save_checkpoint` (full pickle, trusted local files only)."""
    return torch.load(path, map_location=map_location, weights_only=False)
