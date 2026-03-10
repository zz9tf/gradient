#!/bin/bash
# Training script: graddrop. Usage: ./1graddrop.sh [gpu_id] (default gpu_id=0)
GPU_ID="${1:-0}"

source /home/zheng/zheng/miniconda3/etc/profile.d/conda.sh
conda activate gradient
cd /home/zheng/zheng/gradient/mtl_cifar

export CUDA_VISIBLE_DEVICES=$GPU_ID

# baseline
# python train.py \
#     --mode sum \
#     --bs 16 \
#     --epochs 30 \
#     --gpop \
#     --gpop_policy cov_inv \
#     --gpop_beta 0.999 \
#     --monitor \
#     --monitor_detach \
#     --monitor_eps 1e-8 \
#     --monitor_cov_unbiased \
#     --monitor_gpop_beta 0.999 \
#     --monitor_gpop_update \
#     --monitor_gpop_warmup 0 \
#     --monitor_cov_k 1

# python train.py \
#     --mode sum \
#     --bs 16 \
#     --epochs 30 \
#     --gpop \
#     --gpop_policy gg \
#     --gpop_beta 0.999 \
#     --monitor \
#     --monitor_detach \
#     --monitor_eps 1e-8 \
#     --monitor_cov_unbiased \
#     --monitor_gpop_beta 0.999 \
#     --monitor_gpop_update \
#     --monitor_gpop_warmup 0 \
#     --monitor_cov_k 1

# python train.py \
#     --mode sum \
#     --bs 16 \
#     --epochs 30 \
#     --gpop \
#     --gpop_policy cov_mul \
#     --gpop_beta 0.999 \
#     --monitor \
#     --monitor_detach \
#     --monitor_eps 1e-8 \
#     --monitor_cov_unbiased \
#     --monitor_gpop_beta 0.999 \
#     --monitor_gpop_update \
#     --monitor_gpop_warmup 0 \
#     --monitor_cov_k 1

# python train.py \
#     --mode sum \
#     --bs 16 \
#     --epochs 30 \
#     --monitor \
#     --monitor_detach \
#     --monitor_eps 1e-8 \
#     --monitor_cov_unbiased \
#     --monitor_gpop_beta 0.999 \
#     --monitor_gpop_update \
#     --monitor_gpop_warmup 0 \
#     --monitor_cov_k 1 \
#     --rp_monitor


python train.py \
    --mode sum \
    --bs 16 \
    --epochs 30 \
    --monitor \
    --monitor_detach \
    --monitor_eps 1e-8 \
    --monitor_cov_unbiased \
    --monitor_gpop_beta 0.999 \
    --monitor_gpop_update \
    --monitor_gpop_warmup 0 \
    --monitor_cov_k 1 \
    --rp_corr \
    --rp_layer backbone.stage3 \
    --rp_weight inv_sqrt \
    --rp_eps 1e-8 \
    --rp_detach_repr \
    --rp_monitor