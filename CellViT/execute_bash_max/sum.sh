#!/bin/bash
# Training script: execute.sh. Usage: ./execute.sh

GPU_ID="${1:-0}"
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate cellvit
cd /home/zz/zheng/gradient/CellViT

python cell_segmentation/run_cellvit.py --config execute_bash_max/train_config_sum.yaml --gpu $GPU_ID