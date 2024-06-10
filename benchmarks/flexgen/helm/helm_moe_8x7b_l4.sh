python3 helm_run_moe.py --description synthetic_reasoning:model=text,mode=pattern_match --percent 14 86 0 100 0 100 --gpu-batch-size 64 --num-gpu-batches 33 --cpu-cache-compute >> helm_moe_l4_synthetic_14_86_0_100_0_100_64_33_cpu.log 2>&1
python3 helm_run_moe.py --description synthetic_reasoning:model=text,mode=pattern_match --percent 14 86 0 100 0 100 --gpu-batch-size 64 --num-gpu-batches 33 >> helm_moe_l4_synthetic_14_86_0_100_0_100_64_33.log 2>&1

python3 helm_run_moe.py --description summarization_xsum_sampled:model=text,temperature=0.3,device=cpu --percent 12 88 0 100 0 100 --gpu-batch-size 8 --num-gpu-batches 36 --cpu-cache-compute >> helm_moe_l4_summarization_12_88_0_100_0_100_8_36_cpu.log 2>&1 
python3 helm_run_moe.py --description summarization_xsum_sampled:model=text,temperature=0.3,device=cpu --percent 12 88 0 100 0 100 --gpu-batch-size 8 --num-gpu-batches 36 >> helm_moe_l4_summarization_12_88_0_100_0_100_8_36.log 2>&1

python3 helm_run_moe.py --description gsm:model=text_code --percent 15 85 0 100 0 100 --gpu-batch-size 3 --num-gpu-bathces 114 --cpu-cache-compute >> helm_moe_8x7b_l4_gsm_15_85_0_100_0_100_3_114_cpu.log 2>&1
python3 helm_run_moe.py --description gsm:model=text_code --percent 15 85 0 100 0 100 --gpu-batch-size 3 --num-gpu-bathces 114 >> helm_moe_8x7b_l4_gsm_15_85_0_100_0_100_3_114.log 2>&1