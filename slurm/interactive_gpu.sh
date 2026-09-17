#!/usr/bin/env bash
# Select a GPU partition from the current Rangpur sinfo output.
# Usage: bash slurm/interactive_gpu.sh <verified-partition>
set -euo pipefail
PARTITION="${1:-${SLURM_PARTITION:-}}"
if [[ -z "$PARTITION" ]]; then
    echo 'Run sinfo, then: bash slurm/interactive_gpu.sh <GPU-partition>' >&2
    exit 2
fi
# Official Rangpur guidance: testing uses shards, production uses whole GPUs.
# Reserve all four shards for our ML workload rather than crowding other shard users.
if [[ "$PARTITION" == "a100-test" ]]; then
    GRES="shard:4"
    TIME_LIMIT="00:20:00"
else
    GRES="gpu:1"
    TIME_LIMIT="01:00:00"
fi
exec srun --partition="$PARTITION" --gres="$GRES" --cpus-per-task=4 --mem=32G --time="$TIME_LIMIT" --pty bash -l
