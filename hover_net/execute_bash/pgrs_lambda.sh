#!/bin/bash
# Training script: pgrs_lambda. Usage: ./0pgrs_lambda.sh [gpu_id] (default gpu_id=0)
GPU_ID="${1:-0}"

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate hovernet_h100
cd /home/zz/zheng/gradient/hover_net

python run_train.py --gpu "$GPU_ID" --grad-mode pgrs_lambda --grad-beta 0.9999 --grad-tau 0.2 --resume /home/zz/zheng/gradient/hover_net/logs/one_phase/grad_pgrs_lambda_b0.9999_t0.2