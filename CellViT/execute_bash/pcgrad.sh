#!/bin/bash
# Training script: execute.sh. Usage: ./execute.sh

GPU_ID="${1:-0}"
source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate cellvit_env
cd /home/zheng/zheng/gradient/CellViT

python cell_segmentation/run_cellvit.py --config execute_bash/train_config_pcgrad.yaml --gpu $GPU_ID