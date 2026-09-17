#!/usr/bin/env bash
# Submit from the extracted demo package directory.
# sbatch slurm/full_demo.slurm /path/to/OASIS /path/to/existing/comp3710-keras-demo
#SBATCH --job-name=comp3710_demo
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=comp3710_demo_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit using sbatch from the package root}"
OASIS_DATA="${1:?Pass the OASIS data root}"
OLD_PROJECT="${2:?Pass the existing cluster project with .venv and CIFAR results}"
PYTHON="$OLD_PROJECT/.venv/bin/python"
test -x "$PYTHON"
"$PYTHON" -u scripts/collect_demo_evidence.py --cluster \
  --data_root "$OASIS_DATA" \
  --gan_checkpoint results/gan_improved/dcgan.pt \
  --cifar_checkpoint "$OLD_PROJECT/results/part3_dawnbench/resnet18_cifar10.pt" \
  --cifar_data "$OLD_PROJECT/data/cifar10" \
  --output_dir "results/rangpur_demo_${SLURM_JOB_ID}"
