#!/bin/bash
# Part 4 Task 3 - DCGAN brain generation on the OASIS slices (Rangpur).
#
#   sbatch slurm/gan.slurm                        # 128x128, 60 epochs
#   sbatch slurm/gan.slurm --image_size 64        # faster, lower resolution
#
#SBATCH --job-name=oasis_gan
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail
mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate comp3710
[ -f .venv/bin/activate ] && source .venv/bin/activate

export OASIS_ROOT="${OASIS_ROOT:-/home/groups/comp3710/OASIS}"
python part4_recognition/gan/train.py --data_root "$OASIS_ROOT" --image_size 128 --epochs 60 "$@"
