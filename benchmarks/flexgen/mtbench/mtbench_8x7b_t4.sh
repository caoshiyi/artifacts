#!/bin/bash
# python3 mtbench_run_8x7b_moe.py --percent 16 84 0 100 0 100 --gbs 1 --num-gb 146 --cpu-cache-compute >> moe_16_84_0_100_0_100_1_146_cpu.log 2>&1
python3 mtbench_run_8x7b_moe.py --percent 7 93 0 100 0 100 --gbs 32 --num-gb 40 --gen-len 32 --cpu-cache-compute >> moe_gen32_7_93_0_100_0_100_32_40_cpu.log 2>&1
python3 mtbench_run_8x7b_moe.py --percent 8 92 0 100 0 100 --gbs 16 --num-gb 78 --gen-len 64 --cpu-cache-compute >> moe_gen64_8_92_0_100_0_100_16_78_cpu.log 2>&1
python3 mtbench_run_8x7b_moe.py --percent 8 92 0 100 0 100 --gbs 8 --num-gb 139 --gen-len 128 --cpu-cache-compute >> moe_gen128_8_92_0_100_0_100_8_139_cpu.log 2>&1
python3 mtbench_run_8x7b_moe.py --percent 8 92 0 100 0 100 --gbs 8 --num-gb 113 --gen-len 256 --cpu-cache-compute >> moe_gen256_8_92_0_100_0_100_8_113_cpu.log 2>&1

python3 mtbench_run_8x7b_moe.py --percent 7 93 0 100 0 100 --gbs 32 --num-gb 40 --gen-len 32  >> moe_gen32_7_93_0_100_0_100_32_40.log 2>&1
python3 mtbench_run_8x7b_moe.py --percent 8 92 0 100 0 100 --gbs 16 --num-gb 78 --gen-len 64  >> moe_gen64_8_92_0_100_0_100_16_78.log 2>&1
python3 mtbench_run_8x7b_moe.py --percent 8 92 0 100 0 100 --gbs 8 --num-gb 139 --gen-len 128 >> moe_gen128_8_92_0_100_0_100_8_139.log 2>&1
python3 mtbench_run_8x7b_moe.py --percent 8 92 0 100 0 100 --gbs 8 --num-gb 113 --gen-len 256 >> moe_gen256_8_92_0_100_0_100_8_113.log 2>&1