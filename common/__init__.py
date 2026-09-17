"""Shared helpers used by every part of the lab (device selection, seeding, plotting, paths)."""

from common.device import get_device, synchronize
from common.paths import PROJECT_ROOT, RESULTS_DIR, ensure_dir
from common.plotting import save_figure
from common.seed import set_seed

__all__ = [
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "ensure_dir",
    "get_device",
    "save_figure",
    "set_seed",
    "synchronize",
]
