"""Part 1 of 4 - Discrete Fourier Transform (PyTorch version + GPU timing study).

`square_wave`, `square_wave_fourier` and `naive_dft` from the NumPy script are re-implemented with
PyTorch tensor operations. The naive DFT is written as an explicit O(N^2) computation with tensor
ops (the DFT matrix is materialised chunk by chunk and multiplied with the signal) so that it can run
on the GPU *without* using the built-in FFT. Four/five methods are then timed for several signal
sizes N:

    numpy_naive_loop   pure Python double loop                          O(N^2)
    numpy_fft          np.fft.fft                                       O(N log N)
    torch_naive_cpu    vectorised DFT-matrix product on the CPU          O(N^2)
    torch_naive_gpu    vectorised DFT-matrix product on CUDA/MPS         O(N^2), massively parallel
    torch_fft_gpu      torch.fft.fft on the accelerator (reference only)

Run from the repository root:
    python part1_dft/dft_torch.py --sizes 256 512 1024 2048 4096 8192
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable

import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS_DIR, get_device, save_figure, synchronize  # noqa: E402
from common.device import describe_device  # noqa: E402
from part1_dft.square_wave_numpy import naive_dft as naive_dft_numpy  # noqa: E402
from part1_dft.square_wave_numpy import square_wave_fourier as square_wave_fourier_numpy  # noqa: E402

OUT_DIR = RESULTS_DIR / "part1_dft"


# --------------------------------------------------------------------------------------------
# PyTorch re-implementations
# --------------------------------------------------------------------------------------------
def square_wave_torch(t: torch.Tensor, f0: float = 1.0) -> torch.Tensor:
    """Ideal square wave in {-1, +1} using torch ops (works on any device)."""
    return torch.sign(torch.sin(2.0 * math.pi * f0 * t))


def square_wave_fourier_torch(t: torch.Tensor, f0: float, n_harmonics: int) -> torch.Tensor:
    """Fourier-series square wave: all odd harmonics are evaluated at once with broadcasting.

    ``n`` has shape [H, 1] and ``t`` shape [N]; the product broadcasts to [H, N] and is summed over H.
    """
    n = (2 * torch.arange(n_harmonics, device=t.device, dtype=t.dtype) + 1).unsqueeze(1)
    terms = torch.sin(2 * math.pi * n * f0 * t.unsqueeze(0)) / n
    return (4 / math.pi) * terms.sum(dim=0)


def naive_dft_torch(x: torch.Tensor, chunk_elements: int = 1 << 26) -> torch.Tensor:
    """Direct O(N^2) DFT with tensor ops:  X[k] = sum_n x[n] * exp(-2*pi*i*k*n/N).

    Instead of two Python loops the DFT matrix W[k, n] = exp(-2*pi*i*k*n/N) is built explicitly and
    multiplied with x. Real and imaginary parts are handled as two real matrix products so the code
    also runs on Apple MPS (no complex / float64 support there). The matrix is built in row chunks
    of at most ``chunk_elements`` entries to bound memory (the full matrix would be N^2 entries).
    ``(k*n) mod N`` is computed in int64 before converting to float so the phase stays accurate for
    large N even in float32.

    Returns a real tensor of shape [2, N] holding (real, imaginary) parts of X.
    """
    N = x.shape[0]
    device, dtype = x.device, x.dtype
    n = torch.arange(N, device=device, dtype=torch.int64)
    real = torch.empty(N, device=device, dtype=dtype)
    imag = torch.empty(N, device=device, dtype=dtype)
    rows_per_chunk = max(1, min(N, chunk_elements // N))
    for start in range(0, N, rows_per_chunk):
        k = torch.arange(start, min(start + rows_per_chunk, N), device=device, dtype=torch.int64)
        angle = (2.0 * math.pi / N) * torch.remainder(torch.outer(k, n), N).to(dtype)  # [chunk, N]
        real[start : start + len(k)] = torch.cos(angle) @ x
        imag[start : start + len(k)] = -(torch.sin(angle) @ x)
    return torch.stack((real, imag))


def to_numpy_complex(X: torch.Tensor) -> np.ndarray:
    """Convert the [2, N] (real, imag) output of :func:`naive_dft_torch` to a complex NumPy array."""
    X = X.detach().cpu().double()
    return X[0].numpy() + 1j * X[1].numpy()


# --------------------------------------------------------------------------------------------
# Timing utilities
# --------------------------------------------------------------------------------------------
def time_call(fn: Callable[[], object], device: torch.device | None, repeats: int, warmup: int) -> tuple[float, object]:
    """Time ``fn`` and return (best wall-clock seconds, last output).

    GPU work is asynchronous, so we synchronise before starting and after finishing the timer.
    A warm-up call is made first: the very first CUDA call pays for context creation / kernel
    loading which would otherwise dominate the measurement of small problems.
    """
    out = None
    for _ in range(warmup):
        out = fn()
    if device is not None:
        synchronize(device)
    best = math.inf
    for _ in range(repeats):
        if device is not None:
            synchronize(device)
        start = time.perf_counter()
        out = fn()
        if device is not None:
            synchronize(device)
        best = min(best, time.perf_counter() - start)
    return best, out


def relative_error(X: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(X - reference)) / np.max(np.abs(reference)))


def benchmark_size(
    N: int,
    T: float,
    f0: float,
    n_harmonics: int,
    device: torch.device,
    dtype: torch.dtype,
    repeats: int,
    run_numpy_naive: bool,
) -> dict[str, float]:
    """Run every method for one signal length and return a {method: seconds} dictionary."""
    t_np = np.linspace(0.0, T, N, endpoint=False)
    signal_np = square_wave_fourier_numpy(t_np, f0, n_harmonics)
    reference = np.fft.fft(signal_np)
    timings: dict[str, float] = {}

    # -- NumPy ------------------------------------------------------------------------------
    if run_numpy_naive:
        secs, out = time_call(lambda: naive_dft_numpy(signal_np), None, repeats=1, warmup=0)
        timings["numpy_naive_loop"] = secs
        print(f"  numpy_naive_loop : {secs:10.6f} s   rel.err={relative_error(out, reference):.1e}")
    else:
        print("  numpy_naive_loop : skipped (N above --max_naive_n)")
    secs, _ = time_call(lambda: np.fft.fft(signal_np), None, repeats=repeats, warmup=1)
    timings["numpy_fft"] = secs
    print(f"  numpy_fft        : {secs:10.6f} s")

    # -- torch CPU ----------------------------------------------------------------------------
    cpu = torch.device("cpu")
    t_cpu = torch.linspace(0.0, T, N + 1, dtype=torch.float64)[:-1]  # endpoint=False equivalent
    signal_cpu = square_wave_fourier_torch(t_cpu, f0, n_harmonics)
    secs, out = time_call(lambda: naive_dft_torch(signal_cpu), cpu, repeats=repeats, warmup=1)
    timings["torch_naive_cpu"] = secs
    print(f"  torch_naive_cpu  : {secs:10.6f} s   rel.err={relative_error(to_numpy_complex(out), reference):.1e}")

    # -- torch accelerator --------------------------------------------------------------------
    if device.type != "cpu":
        t_dev = torch.linspace(0.0, T, N + 1, dtype=dtype, device=device)[:-1]
        signal_dev = square_wave_fourier_torch(t_dev, f0, n_harmonics)
        secs, out = time_call(lambda: naive_dft_torch(signal_dev), device, repeats=repeats, warmup=1)
        timings["torch_naive_gpu"] = secs
        print(f"  torch_naive_gpu  : {secs:10.6f} s   rel.err={relative_error(to_numpy_complex(out), reference):.1e}")
        if device.type == "cuda":  # torch.fft is not implemented for MPS
            secs, _ = time_call(lambda: torch.fft.fft(signal_dev), device, repeats=repeats, warmup=1)
            timings["torch_fft_gpu"] = secs
            print(f"  torch_fft_gpu    : {secs:10.6f} s   (built-in, reference only)")
    else:
        print("  torch_naive_gpu  : no CUDA/MPS device available on this machine")

    order = sorted(timings, key=timings.get)
    print("  fastest -> slowest: " + " < ".join(order))
    return timings


def plot_timings(results: dict[int, dict[str, float]], device_name: str) -> None:
    methods = sorted({m for r in results.values() for m in r})
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        sizes = [N for N in sorted(results) if method in results[N]]
        ax.plot(sizes, [results[N][method] for N in sizes], marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("signal length N")
    ax.set_ylabel("best wall-clock time (s)")
    ax.set_title(f"DFT timing vs. N  (accelerator: {device_name})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    save_figure(fig, OUT_DIR / "dft_timings.png")


def save_csv(results: dict[int, dict[str, float]]) -> None:
    methods = sorted({m for r in results.values() for m in r})
    path = OUT_DIR / "dft_timings.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", *methods])
        for N in sorted(results):
            writer.writerow([N, *[f"{results[N][m]:.6g}" if m in results[N] else "" for m in methods]])
    print(f"[saved] {path}")


# --------------------------------------------------------------------------------------------
def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--max_naive_n", type=int, default=2048, help="skip the pure-Python DFT above this N")
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--f0", type=float, default=1.0)
    parser.add_argument("--harmonics", type=int, default=50, help="harmonics used to build the test signal")
    parser.add_argument("--repeats", type=int, default=5, help="timed repetitions (best time is reported)")
    parser.add_argument("--device", default=None, help="force a device, e.g. cuda, mps or cpu")
    parser.add_argument("--output_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.output_dir

    device = get_device(args.device)
    # MPS has no float64 support; CUDA/CPU use float64 to match NumPy's precision.
    dtype = torch.float32 if device.type == "mps" else torch.float64
    print(f"accelerator: {describe_device(device)}  dtype: {dtype}\n")

    # Sanity check that the torch re-implementations agree with the NumPy originals.
    t_np = np.linspace(0.0, args.T, 2048, endpoint=False)
    t_th = torch.linspace(0.0, args.T, 2049, dtype=torch.float64)[:-1]
    assert np.allclose(square_wave_fourier_numpy(t_np, args.f0, 5), square_wave_fourier_torch(t_th, args.f0, 5).numpy())
    assert np.allclose(np.sign(np.sin(2 * np.pi * args.f0 * t_np)), square_wave_torch(t_th, args.f0).numpy())
    print("torch square_wave / square_wave_fourier match the NumPy versions: OK\n")

    results: dict[int, dict[str, float]] = {}
    for N in args.sizes:
        print(f"N = {N}")
        results[N] = benchmark_size(
            N, args.T, args.f0, args.harmonics, device, dtype, args.repeats, run_numpy_naive=N <= args.max_naive_n
        )
        print()

    save_csv(results)
    plot_timings(results, describe_device(device))


if __name__ == "__main__":
    main()
