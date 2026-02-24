#!/bin/bash
# Training script: pgrs. Usage: ./0pgrs.sh [gpu_id] (default gpu_id=0)
GPU_ID="${1:-0}"

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate hovernet_h100
cd /home/zz/zheng/gradient/hover_net

python run_train.py --gpu "$GPU_ID" --grad-mode pgrs --grad-beta 0.99 --grad-tau 0.2 --resume='/home/zz/zheng/gradient/hover_net/logs/grad_pgrs_b0.99_t0.2_gw1'