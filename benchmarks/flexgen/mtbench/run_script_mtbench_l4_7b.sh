#!/bin/bash
# python3 mtbench_run_moe.py --percent 16 84 0 100 0 100 --gbs 1 --num-gb 146 --cpu-cache-compute >> moe_16_84_0_100_0_100_1_146_cpu.log 2>&1
python3 mtbench_run_moe.py --percent 12 88 0 100 0 100 --gbs 64 --num-gb 21 --gen-len 32 --cpu-cache-compute >> moe_gen32_12_88_0_100_0_100_64_21_cpu.log 2>&1
python3 mtbench_run_moe.py --percent 12 88 0 100 0 100 --gbs 64 --num-gb 20 --gen-len 64 --cpu-cache-compute >> moe_gen64_12_88_0_100_0_100_64_20_cpu.log 2>&1
python3 mtbench_run_moe.py --percent 12 88 0 100 0 100 --gbs 64 --num-gb 18 --gen-len 128 --cpu-cache-compute >> moe_gen128_12_88_0_100_0_100_64_18_cpu.log 2>&1
python3 mtbench_run_moe.py --percent 14 86 0 100 0 100 --gbs 32 --num-gb 30 --gen-len 256 --cpu-cache-compute >> moe_gen256_14_86_0_100_0_100_32_30_cpu.log 2>&1

python3 mtbench_run_moe.py --percent 12 88 0 100 0 100 --gbs 64 --num-gb 21 --gen-len 32 >> moe_gen32_12_88_0_100_0_100_64_21.log 2>&1
python3 mtbench_run_moe.py --percent 12 88 0 100 0 100 --gbs 64 --num-gb 20 --gen-len 64 >> moe_gen64_12_88_0_100_0_100_64_20.log 2>&1
python3 mtbench_run_moe.py --percent 12 88 0 100 0 100 --gbs 64 --num-gb 18 --gen-len 128 >> moe_gen128_12_88_0_100_0_100_64_18.log 2>&1
python3 mtbench_run_moe.py --percent 14 86 0 100 0 100 --gbs 32 --num-gb 30 --gen-len 256 >> moe_gen256_14_86_0_100_0_100_32_30.log 2>&1