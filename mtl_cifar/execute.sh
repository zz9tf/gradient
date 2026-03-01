#!/bin/bash
# Training script: graddrop. Usage: ./1graddrop.sh [gpu_id] (default gpu_id=0)
GPU_ID="${1:-0}"

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate hovernet_h100
cd /home/zz/zheng/gradient/mtl_cifar

python train.py --gpu_id $GPU_ID --grad_mode sum