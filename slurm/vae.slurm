#!/bin/bash
# Part 4 Task 1 - VAE on the OASIS slices (Rangpur).
#
#   sbatch slurm/vae.slurm                              # latent_dim = 2 (manifold grid)
#   sbatch slurm/vae.slurm --latent_dim 32 --epochs 30  # higher-dimensional latent + UMAP
#
#SBATCH --job-name=oasis_vae
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail
mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate comp3710
[ -f .venv/bin/activate ] && source .venv/bin/activate

export OASIS_ROOT="${OASIS_ROOT:-/home/groups/comp3710/OASIS}"
python part4_recognition/vae/train.py --data_root "$OASIS_ROOT" --image_size 128 --epochs 20 "$@"
