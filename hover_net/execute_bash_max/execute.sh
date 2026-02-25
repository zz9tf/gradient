#!/bin/bash
script_path="/home/zz/zheng/gradient/hover_net/execute_bash_max"
# task run "hovernet_pgrs" "$script_path/pgrs.sh 1"
# task run "hovernet_pgrs_keep1" "$script_path/pgrs_keep1.sh 4"
# task run "pgrs_stage" "$script_path/pgrs_stage.sh 2"

# task run "sum" "$script_path/sum.sh 0"
# task run "graddrop" "$script_path/graddrop.sh 0"
task run "pcgrad" "$script_path/pcgrad.sh 0"
# task run "pgrs_common_gate" "$script_path/pgrs_common_gate.sh 1"
# task run "pgrs_common_gate_-0.012" "$script_path/pgrs_common_gate_-0.012.sh 0"
# task run "pgrs_common_gate_-0.01" "$script_path/pgrs_common_gate_-0.01.sh 0"
