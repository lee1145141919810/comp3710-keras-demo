#!/bin/bash
# Part 4 Task 2 - UNet segmentation of the OASIS slices (Rangpur).
#
#   sbatch slurm/unet.slurm                     # train + evaluate on the test split
#
# Inference only (e.g. live during the demo, on an interactive GPU node):
#   python part4_recognition/unet/predict.py --checkpoint results/part4_unet/unet_best.pt --data_root $OASIS_ROOT
#
#SBATCH --job-name=oasis_unet
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail
mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate comp3710
[ -f .venv/bin/activate ] && source .venv/bin/activate

export OASIS_ROOT="${OASIS_ROOT:-/home/groups/comp3710/OASIS}"
python part4_recognition/unet/train.py --data_root "$OASIS_ROOT" --epochs 15 --batch_size 16 --amp --augment "$@"
