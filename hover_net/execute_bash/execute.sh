#!/bin/bash
script_path="/home/zz/zheng/gradient/hover_net/execute_bash"
task run "pgrs_lambda" "$script_path/pgrs_lambda.sh 4"
task run "pgrs" "$script_path/pgrs.sh 1"
# task run "sum" "$script_path/sum.sh 4"
# task run "graddrop" "$script_path/graddrop.sh 0"
# task run "pcgrad" "$script_path/pcgrad.sh 1"
