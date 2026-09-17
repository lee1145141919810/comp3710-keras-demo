#!/usr/bin/env bash
# Execute inside an allocated Rangpur GPU job, from any working directory.
# Arguments: trained CIFAR checkpoint, trained UNet checkpoint, OASIS root.
set -euo pipefail
cd "$(dirname "$0")/.."
CIFAR_CHECKPOINT="${1:?Pass the trained ResNet checkpoint}"
UNET_CHECKPOINT="${2:?Pass the trained UNet checkpoint}"
OASIS_DATA="${3:?Pass the OASIS PNG root}"
PYTHON="${PYTHON:-python}"
RUN_DIR="results/live_demo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
exec > >(tee "$RUN_DIR/demo.log") 2>&1
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "This demo requires an allocated CUDA GPU"; print(torch.cuda.get_device_name(0))'
"$PYTHON" part3_cnn/dawnbench/train_cifar10.py --device cuda --no_download --eval_only --checkpoint "$CIFAR_CHECKPOINT" --output_dir "$RUN_DIR/cifar_inference"
"$PYTHON" part3_cnn/dawnbench/train_cifar10.py --device cuda --no_download --epochs 1 --checkpoint "$CIFAR_CHECKPOINT" --output_dir "$RUN_DIR/cifar_epoch"
"$PYTHON" part4_recognition/unet/predict.py --device cuda --checkpoint "$UNET_CHECKPOINT" --data_root "$OASIS_DATA" --output_dir "$RUN_DIR/unet"
