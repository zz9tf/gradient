#!/bin/bash
# Training script: pcgrad. Usage: ./2pcgrad.sh [gpu_id] (default gpu_id=0)
GPU_ID="${1:-0}"

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate hovernet_h100
cd /home/zz/zheng/gradient/hover_net

python run_train.py --gpu "$GPU_ID" --grad-mode pcgrad