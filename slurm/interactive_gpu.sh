#!/usr/bin/env bash
# Select a GPU partition from the current Rangpur sinfo output.
# Usage: bash slurm/interactive_gpu.sh <verified-partition>
set -euo pipefail
PARTITION="${1:-${SLURM_PARTITION:-}}"
if [[ -z "$PARTITION" ]]; then
    echo 'Run sinfo, then: bash slurm/interactive_gpu.sh <GPU-partition>' >&2
    exit 2
fi
exec srun --partition="$PARTITION" --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00 --pty bash -l
