#!/bin/bash
# Training script: execute.sh. Usage: ./execute.sh

set -euo pipefail

GPU_ID="${1:-0}"
source "${HOME}/zheng/miniconda3/etc/profile.d/conda.sh"
conda activate cellvit_env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLVIT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CELLVIT_ROOT}"

python cell_segmentation/run_cellvit.py --config execute_bash/train_config_graddrop.yaml --gpu "${GPU_ID}"