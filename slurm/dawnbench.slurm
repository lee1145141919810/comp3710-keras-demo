#!/bin/bash
# Part 3.2 - DAWNBench ResNet-18 / CIFAR-10 on the UQ Rangpur cluster.
#
#   sbatch slurm/dawnbench.slurm                 # full 30-epoch run (target >= 94 %)
#   sbatch slurm/dawnbench.slurm --epochs 1      # single-epoch demo (extra args are forwarded)
#
# CIFAR-10 must already be under data/cifar10 (compute nodes may have no internet):
#   python -c "import torchvision; torchvision.datasets.CIFAR10('data/cifar10', download=True)"
#
#SBATCH --job-name=dawnbench
#SBATCH --partition=a100          # check available partitions with `sinfo`
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

set -euo pipefail
mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# --- environment: adapt ONE of these to your own setup ---------------------------------------
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate comp3710
[ -f .venv/bin/activate ] && source .venv/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python part3_cnn/dawnbench/train_cifar10.py --no_download --tta "$@"
