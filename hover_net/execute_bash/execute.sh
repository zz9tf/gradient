#!/bin/bash
script_path="/home/zheng/zheng/gradient/hover_net/execute_bash"

task run "sum" "$script_path/sum.sh 0"
task run "pcgrad" "$script_path/pcgrad.sh 1"
# task run "graddrop" "$script_path/graddrop.sh 0"
