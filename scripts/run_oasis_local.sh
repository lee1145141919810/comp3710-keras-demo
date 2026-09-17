#!/usr/bin/env bash
# Train Part 4 Medium (VAE, UNet) on a local copy of keras_png_slices_data.
# Default path is the one used on the author's Mac; override with OASIS_ROOT.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

ROOT="${OASIS_ROOT:-${HOME}/Downloads/keras_png_slices_data}"
python part4_recognition/oasis_data.py --data_root "$ROOT"

echo "== Task 1: VAE (2-D latent, 128 px, 20 epochs) =="
python part4_recognition/vae/train.py --data_root "$ROOT" --image_size 128 --latent_dim 2 --epochs 20

echo "== Task 2: UNet segmentation =="
python part4_recognition/unet/train.py --data_root "$ROOT" --epochs 15 --batch_size 8 --augment

echo "== Task 2: live inference (demo command) =="
python part4_recognition/unet/predict.py --checkpoint results/part4_unet/unet_best.pt --data_root "$ROOT"

# Hard-only task: opt in explicitly with RUN_GAN=1.
if [[ "${RUN_GAN:-0}" == "1" ]]; then
    python part4_recognition/gan/train.py --data_root "$ROOT" --image_size 128 --epochs 60
fi
