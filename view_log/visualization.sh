#!/bin/bash

source /home/zheng/zheng/miniconda3/etc/profile.d/conda.sh
conda activate gradient
cd /home/zheng/zheng/gradient/view_log
input_path="${1:-0}"

python plot_all.py \
    --input ${input_path}

python plot_repr.py \
    --input ${input_path}