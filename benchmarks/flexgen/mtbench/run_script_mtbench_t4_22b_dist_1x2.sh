#!/bin/bash
# python3 mtbench_run_moe_dist.py --percent 1 99 0 100 0 100 --gbs 16 --num-gb 46 --gen-len 32 --cpu-cache-compute >> moe_dist_gen32_1_99_0_100_0_100_16_46_cpu.log 2>&1
# python3 mtbench_run_moe_dist.py --percent 1 99 0 100 0 100 --gbs 8 --num-gb 90 --gen-len 64 --cpu-cache-compute >> moe_dist_gen64_1_99_0_100_0_100_8_90_cpu.log 2>&1
# python3 mtbench_run_moe_dist.py --percent 1 99 0 100 0 100 --gbs 3 --num-gb 214 --gen-len 128 --cpu-cache-compute >> moe_dist_gen128_1_99_0_100_0_100_3_214_cpu.log 2>&1
# python3 mtbench_run_moe_dist.py --percent 1 99 0 100 0 100 --gbs 4 --num-gb 130 --gen-len 256 --cpu-cache-compute >> moe_dist_gen256_1_99_0_100_0_100_4_130_cpu.log 2>&1 

# python3 mtbench_run_moe_dist.py --percent 1 99 0 100 0 100 --gbs 16 --num-gb 46 --gen-len 32 >> moe_dist_gen32_1_99_0_100_0_100_16_46.log 2>&1
# python3 mtbench_run_moe_dist.py --percent 1 99 0 100 0 100 --gbs 8 --num-gb 90 --gen-len 64 >> moe_dist_gen64_1_99_0_100_0_100_8_90.log 2>&1
# python3 mtbench_run_moe_dist.py --percent 1 99 0 100 0 100 --gbs 3 --num-gb 214 --gen-len 128 >> moe_dist_gen128_1_99_0_100_0_100_3_214.log 2>&1
# python3 mtbench_run_moe_dist.py --percent 1 99 0 100 0 100 --gbs 4 --num-gb 130 --gen-len 256 >> moe_dist_gen256_1_99_0_100_0_100_4_130.log 2>&1

#!/bin/bash

MY_IPADDR=$(hostname -i)
all_hosts=$MY_IPADDR
N_GPUS=2
N_CORES_PER_GPU=16

PYTHON_EXEC=$CONDA_PREFIX/bin/python
PYTHON_SCRIPT=mtbench_run_moe_dist.py

pgrep -fl python | awk '!/mtbench_run_moe_dist\.py/{print $1}' | xargs sudo kill

set -x

echo "MY_IPADDR: $MY_IPADDR"

run_command() {
  local percent=$1
  local gbs=$2
  local num_gb=$3
  local gen_len=$4
  local cpu_cache_compute=$5
  local log_file=$6

  if [ "$cpu_cache_compute" == "true" ]; then
    mpirun \
      --mca btl_tcp_if_exclude lo,docker0 \
      --mca oob_tcp_if_exclude lo,docker0 \
      --map-by ppr:$N_GPUS:node:pe=$N_CORES_PER_GPU --oversubscribe -H $all_hosts \
      --bind-to core -x OMP_NUM_THREADS=$N_CORES_PER_GPU \
      $PYTHON_EXEC $PYTHON_SCRIPT \
        --head-ip $MY_IPADDR \
        --port 7777 \
        --use-mpi \
        --model mistralai/Mixtral-8x22B-Instruct-v0.1 \
        --gbs $gbs \
        --num-gb $num_gb \
        --gen-len $gen_len \
        --percent $percent \
        --comm-device cpu \
        --cpu-cache-compute \
        --no-log >> $log_file 2>&1
  else
    mpirun \
      --mca btl_tcp_if_exclude lo,docker0 \
      --mca oob_tcp_if_exclude lo,docker0 \
      --map-by ppr:$N_GPUS:node:pe=$N_CORES_PER_GPU --oversubscribe -H $all_hosts \
      --bind-to core -x OMP_NUM_THREADS=$N_CORES_PER_GPU \
      $PYTHON_EXEC $PYTHON_SCRIPT \
        --head-ip $MY_IPADDR \
        --port 7777 \
        --use-mpi \
        --model mistralai/Mixtral-8x22B-Instruct-v0.1 \
        --gbs $gbs \
        --num-gb $num_gb \
        --gen-len $gen_len \
        --percent $percent \
        --comm-device cpu \
        --no-log >> $log_file 2>&1
  fi
}

run_command "1 99 0 100 0 100" 4 182 64 true "moe_dist_gen64_1_99_0_100_0_100_4_182_cpu_1x2.log"
run_command "1 99 0 100 0 100" 4 182 64 false "moe_dist_gen64_1_99_0_100_0_100_4_182_1x2.log"

run_command "1 99 0 100 0 100" 8 80 128 true "moe_dist_gen128_1_99_0_100_0_100_8_80_cpu_1x2.log"
run_command "1 99 0 100 0 100" 8 80 128 false "moe_dist_gen128_1_99_0_100_0_100_8_80_1x2.log"

run_command "1 99 0 100 0 100" 2 261 256 true "moe_dist_gen256_1_99_0_100_0_100_2_261_cpu_1x2.log"
run_command "1 99 0 100 0 100" 2 261 256 false "moe_dist_gen256_1_99_0_100_0_100_2_261_1x2.log"

# NOTE: fail runs below
run_command "1 99 0 100 0 100" 16 46 32 true "moe_dist_gen32_1_99_0_100_0_100_16_46_cpu_1x2.log"
run_command "1 99 0 100 0 100" 16 46 32 false "moe_dist_gen32_1_99_0_100_0_100_16_46_1x2.log"