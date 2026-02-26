#!/bin/bash
# Training script: pcgrad. Usage: ./2pcgrad.sh [gpu_id] (default gpu_id=0)

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate hovernet_h100

cd /home/zz/zheng/gradient 
python '/home/zz/zheng/gradient/hover_net/run_test_summary.py' \
    '/home/zz/zheng/gradient/hover_net/logs/valid_one' \
    --gpu=2 \
    --max-epoch=40 \
    --force=1 \
    --csv='/home/zz/zheng/gradient/hover_net/logs/valid_one/summary.csv'