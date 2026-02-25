#!/bin/bash
# Training script: execute.sh. Usage: ./execute.sh

source ~/zheng/miniconda3/etc/profile.d/conda.sh
conda activate cellvit_env
cd /home/zheng/zheng/gradient/CellViT

cuda_device=0

python cell_segmentation/run_cellvit.py --config execute_bash/train_config_pcgrad.yaml --gpu $cuda_device