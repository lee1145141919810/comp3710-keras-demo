"""Part 1 of 4 - Discrete Fourier Transform (NumPy version).

1. Reconstruct a square wave from its Fourier series using an increasing number of odd harmonics
   (1, 3, 5, 20, 50) and plot the approximations against the ideal square wave.
2. Decompose the reconstructed wave back into its harmonics with a naive O(N^2) DFT and with
   NumPy's FFT, timing both, and compare the recovered spectrum with the harmonics used to build it.

Run from the repository root:
    python part1_dft/square_wave_numpy.py --harmonics 1 3 5 20 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable

import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS_DIR, save_figure  # noqa: E402

OUT_DIR = RESULTS_DIR / "part1_dft"


# --------------------------------------------------------------------------------------------
# Signal construction
# --------------------------------------------------------------------------------------------
def square_wave(t: np.ndarray, f0: float = 1.0) -> np.ndarray:
    """Ideal square wave with fundamental frequency ``f0`` taking values in {-1, +1}."""
    return np.sign(np.sin(2.0 * np.pi * f0 * t))


def square_wave_fourier(t: np.ndarray, f0: float, n_harmonics: int) -> np.ndarray:
    """Fourier-series approximation of a square wave using the first ``n_harmonics`` odd harmonics.

    A square wave only contains odd harmonics with amplitude 4 / (pi * n):
        x(t) = 4/pi * sum_{k=0}^{N-1} sin(2*pi*(2k+1)*f0*t) / (2k+1)
    """
    result = np.zeros_like(t)
    for k in range(n_harmonics):
        n = 2 * k + 1
        result += np.sin(2 * np.pi * n * f0 * t) / n
    return (4 / np.pi) * result


# --------------------------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------------------------
def naive_dft(x: np.ndarray) -> np.ndarray:
    """Textbook O(N^2) DFT:  X[k] = sum_n x[n] * exp(-2j*pi*k*n/N).

    The double Python loop is intentionally kept to show the cost of the direct formula.
    """
    N = len(x)
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


def magnitude_spectrum(X: np.ndarray, N: int, T: float) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided frequency axis and amplitude-normalised magnitude (2/N * |X|)."""
    xf = np.fft.fftfreq(N, d=T / N)[: N // 2]
    magnitude = 2.0 / N * np.abs(X[: N // 2])
    return xf, magnitude


# --------------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------------
def plot_reconstructions(t: np.ndarray, square: np.ndarray, f0: float, harmonics: list[int]) -> None:
    n_panels = len(harmonics) + 1
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = axes.ravel()

    axes[0].plot(t, square, "k", label="Square wave")
    axes[0].set_title("Original square wave")

    for ax, nh in zip(axes[1:], harmonics):
        y = square_wave_fourier(t, f0, nh)
        ax.plot(t, y, label=f"N={nh} harmonics")
        ax.plot(t, square, "k--", alpha=0.5, label="Square wave")
        ax.set_title(f"Fourier approximation, N={nh}")

    for ax in axes[:n_panels]:
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)
    for ax in axes[n_panels:]:
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "square_wave_reconstruction.png")


def plot_spectrum(t: np.ndarray, signal: np.ndarray, X: np.ndarray, T: float, title: str, filename: str) -> None:
    N = len(signal)
    xf, magnitude = magnitude_spectrum(X, N, T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(t, signal, color="c")
    ax1.set_title(title)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, T)
    ax1.grid(True)

    ax2.stem(xf, magnitude, basefmt=" ")
    ax2.set_title("DFT magnitude spectrum (2/N * |X[k]|)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_xlim(0, 50)
    ax2.grid(True)
    for i in range(1, 20, 2):  # mark the odd harmonics used to build the wave
        ax2.axvline(xf[i], color="r", linestyle="--", alpha=0.5, label=f"{i}*f0 = {xf[i]:.1f} Hz" if i <= 5 else None)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, OUT_DIR / filename)


def report_recovered_harmonics(X: np.ndarray, N: int, T: float, f0: float, n_report: int = 6) -> None:
    """Compare DFT amplitudes at odd harmonics with the theoretical 4/(pi*n) Fourier coefficients."""
    xf, magnitude = magnitude_spectrum(X, N, T)
    print(f"{'harmonic n':>10} {'freq (Hz)':>10} {'DFT amp':>10} {'4/(pi n)':>10} {'abs err':>10}")
    for k in range(n_report):
        n = 2 * k + 1
        idx = int(round(n * f0 * T))  # frequency resolution is 1/T Hz per bin
        theory = 4 / (np.pi * n)
        print(f"{n:>10d} {xf[idx]:>10.2f} {magnitude[idx]:>10.4f} {theory:>10.4f} {abs(magnitude[idx] - theory):>10.2e}")
    even_energy = magnitude[2::2][: N // 4].max()
    print(f"largest even-harmonic magnitude (should be ~0): {even_energy:.2e}")


# --------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--N", type=int, default=2048, help="number of sample points")
    parser.add_argument("--T", type=float, default=1.0, help="signal duration in seconds")
    parser.add_argument("--f0", type=float, default=1.0, help="fundamental frequency in Hz")
    parser.add_argument("--harmonics", type=int, nargs="+", default=[1, 3, 5, 20, 50], help="harmonic counts to plot")
    parser.add_argument("--dft_harmonics", type=int, default=50, help="harmonics in the wave fed to the DFT")
    args = parser.parse_args()

    t = np.linspace(0.0, args.T, args.N, endpoint=False)  # endpoint=False: the interval is periodic
    square = square_wave(t, args.f0)

    # 1. Reconstruction with a growing number of harmonics -------------------------------------
    plot_reconstructions(t, square, args.f0, args.harmonics)
    for nh in args.harmonics:
        y = square_wave_fourier(t, args.f0, nh)
        err = np.abs(y - square)
        # Gibbs phenomenon: the overshoot next to a jump tends to ~9 % and never vanishes, while the
        # mean error keeps shrinking as more harmonics are added.
        print(f"N={nh:>3d} harmonics: mean |error|={err.mean():.4f}, peak overshoot={100 * (np.abs(y).max() - 1):.1f}%")

    # 2. Decompose the reconstructed wave with the naive DFT and the FFT ------------------------
    signal = square_wave_fourier(t, args.f0, args.dft_harmonics)

    start = time.perf_counter()
    dft_result = naive_dft(signal)
    naive_duration = time.perf_counter() - start

    start = time.perf_counter()
    fft_result = np.fft.fft(signal)
    fft_duration = time.perf_counter() - start

    print("\n--- DFT / FFT performance comparison (NumPy) ---")
    print(f"Naive DFT execution time : {naive_duration:.6f} s")
    print(f"NumPy FFT execution time : {fft_duration:.6f} s")
    if fft_duration > 0:
        print(f"FFT is approximately {naive_duration / fft_duration:.1f}x faster.")
    print(f"Naive DFT matches NumPy FFT (np.allclose): {np.allclose(dft_result, fft_result)}")

    print(f"\nHarmonics recovered from the {args.dft_harmonics}-harmonic reconstruction:")
    report_recovered_harmonics(dft_result, args.N, args.T, args.f0)
    plot_spectrum(t, signal, dft_result, args.T, f"Square wave built from {args.dft_harmonics} harmonics", "dft_spectrum_reconstructed.png")

    # The ideal square wave has infinite bandwidth: harmonics above Nyquist alias onto lower bins.
    print("\nHarmonics recovered from the IDEAL square wave (np.sign):")
    ideal_fft = np.fft.fft(square)
    report_recovered_harmonics(ideal_fft, args.N, args.T, args.f0)
    plot_spectrum(t, square, ideal_fft, args.T, "Ideal square wave (np.sign)", "dft_spectrum_ideal.png")


if __name__ == "__main__":
    main()
