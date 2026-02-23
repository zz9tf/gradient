#!/bin/bash
# Training script: execute.sh. Usage: ./execute.sh

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate cellvit
cd /home/zz/zheng/gradient/CellViT

cuda_device=2

python cell_segmentation/run_cellvit.py --config train_config_max.yaml --gpu $cuda_device