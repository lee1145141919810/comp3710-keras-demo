#!/usr/bin/env bash
# Read-only environment check. Run on Rangpur after logging in / allocating a GPU.
set -euo pipefail
MODE="${1:-login}"
if [[ "$MODE" != "login" && "$MODE" != "gpu" ]]; then
    echo 'Usage: bash scripts/cluster_preflight.sh [login|gpu]' >&2
    exit 2
fi
printf 'Host: '; hostname
printf 'Working directory: '; pwd
if [[ "$MODE" == "login" ]]; then
    command -v sinfo >/dev/null || { echo 'sinfo is unavailable: check the login host/environment.' >&2; exit 1; }
    sinfo -o '%P %a %l %G'
    squeue --me
else
    [[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Allocate a SLURM GPU job before running this check.' >&2; exit 1; }
    printf 'SLURM job: %s\n' "$SLURM_JOB_ID"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    "${PYTHON:-python}" - <<'PY'
import sys
import torch
import torchvision
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA in this allocated job'
x = torch.ones(16, device='cuda')
assert x.sum().item() == 16
print('CUDA tensor execution: OK')
print('GPU:', torch.cuda.get_device_name(0))
print('bf16 supported:', torch.cuda.is_bf16_supported())
PY
fi
