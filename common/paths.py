"""Project-relative paths so every script writes its outputs to the same place."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
