python3 helm_run_moe.py --description synthetic_reasoning:model=text,mode=pattern_match --percent 7 93 0 100 0 100 --gpu-batch-size 32 --num-gpu-batches 61 --cpu-cache-compute >> helm_moe_t4_synthetic_reasoning_7_93_0_100_0_100_32_61_cpu.log 2>&1
python3 helm_run_moe.py --description synthetic_reasoning:model=text,mode=pattern_match --percent 7 93 0 100 0 100 --gpu-batch-size 32 --num-gpu-batches 61 >> helm_moe_t4_synthetic_reasoning_7_93_0_100_0_100_32_61.log 2>&1

python3 helm_run_moe.py --description summarization_xsum_sampled:model=text,temperature=0.3,device=cpu --percent 7 93 0 100 0 100 --gpu-batch-size 3 --num-gpu-batches 92 --cpu-cache-compute >> helm_moe_t4_summarization_xsum_sampled_7_93_0_100_0_100_3_92_cpu.log 2>&1
python3 helm_run_moe.py --description summarization_xsum_sampled:model=text,temperature=0.3,device=cpu --percent 7 93 0 100 0 100 --gpu-batch-size 3 --num-gpu-batches 92 >> helm_moe_t4_summarization_xsum_sampled_7_93_0_100_0_100_3_92.log 2>&1

python3 helm_run_moe.py --description gsm:model=text_code --percent 8 92 0 100 0 100 --gpu-batch-size 1 --num-gpu-batches 315 --cpu-cache-compute >> helm_moe_8x7b_t4_gsm_8_92_0_100_0_100_1_315_cpu.log 2>&1
python3 helm_run_moe.py --description gsm:model=text_code --percent 8 92 0 100 0 100 --gpu-batch-size 1 --num-gpu-batches 315 >> helm_moe_8x7b_t4_gsm_8_92_0_100_0_100_1_315.log 2>&1