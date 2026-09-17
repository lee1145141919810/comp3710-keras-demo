"""Device selection that works on the Rangpur cluster (CUDA), Apple Silicon (MPS) and plain CPUs."""

from __future__ import annotations

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """Return the best available torch device.

    Args:
        preferred: Optional explicit device string (``"cuda"``, ``"cuda:1"``, ``"mps"``, ``"cpu"``).
            When given it is used verbatim; otherwise CUDA > MPS > CPU is picked automatically.
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Block until all queued kernels on ``device`` have finished.

    GPU kernels are launched asynchronously, so wall-clock timing is only meaningful after a
    synchronisation point. CPU devices need nothing.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def describe_device(device: torch.device) -> str:
    """Human readable device name for logs (e.g. ``cuda (NVIDIA A100-SXM4-80GB)``)."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return device.type
