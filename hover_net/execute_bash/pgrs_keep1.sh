#!/bin/bash
# Training script: pgrs_lpf1. Usage: ./0pgrs_lpf1.sh [gpu_id] (default gpu_id=0)
# --grad-gpop-key: np | hv | tp
GPU_ID="${1:-0}"

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate hovernet_h100
cd /home/zz/zheng/gradient/hover_net

python run_train.py --gpu "$GPU_ID" --grad-mode pgrs_lpf1 --grad-beta 0.99 --grad-tau 0.2 --grad-gpop-key np --grad-gpop-use-weight 1 --resume='/home/zz/zheng/gradient/hover_net/logs/grad_pgrs_lpf1_b0.99_t0.2_gknp_gw1'