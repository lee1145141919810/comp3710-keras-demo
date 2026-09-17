#!/usr/bin/env bash
# Run the CPU-friendly parts of the lab (1, 2, 3.1) from the repository root.
# Parts 3.2 and 4 need a GPU + OASIS / CIFAR-10; see slurm/ and README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

echo "== Part 1: square-wave reconstruction + naive DFT vs FFT =="
python part1_dft/square_wave_numpy.py
echo "== Part 1: PyTorch DFT timing sweep =="
python part1_dft/dft_torch.py --sizes 256 512 1024 2048 --repeats 3
echo "== Part 2: eigenfaces + random forest =="
python part2_eigenfaces/eigenfaces.py
echo "== Part 3.1: LFW CNN =="
python part3_cnn/lfw_cnn.py --epochs 40
echo "Done. Figures and JSON metrics are under results/."
