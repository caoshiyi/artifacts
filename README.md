# MoE-Lightning: Artifact Evaluation

This document contains instruction for ASPLOS 2025 artifact evaluation for the paper *MoE-Lightning: High-Throughput MoE Inference on
Memory-constrained GPUs*. 

## Artifact Overview 
In this artifact, we propose a high-throughput MoE batch infer-ence system, MoE-Lightning, that significantly outperforms
SOTA baselines. MoE-Lightning introduces a novel CPU-GPU-I/O pipelining schedule, *CGOPipe*, with paged weights to achieve
high resource utilization, and a performance model, *HRM*, based on a Hierarchical Roofline Model we introduce to help
find policies with higher throughput than existing systems.


## Installation 
```
git clone -b asplos-artifact https://github.com/caoshiyi/FastMoE.git 
cd FastMoE
conda create -n fastmoe python=3.11
conda activate fastmoe
conda install -c https://software.repos.intel.com/python/conda/ -c conda-forge onemkl-sycl-blas
conda install -c https://software.repos.intel.com/python/conda/ -c conda-forge mkl-include
pip install -e .
pip install triton==2.2.0
```

## Run tests (MTBench)
The first-time weights loading can take ~10min. Adjust `--gen-len` for different generation configurations.
S2 (Recommend using GCP g2-standard-48 instance)
```
python -m fastmoe.serve.launch_server --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 --port 30000 --cpu-mem-bdw 76 --avg-prompt-len 77 --gen-len 32
cd benchmarks/mtbench
python bench.py --port 30000 --max-new-tokens 32 --ubs 324 --n-ub 14
```
The corresponding `--ubs` and `--n-ub` can be obtained from the optimizer results after launching the serve:
```
CostModelConfig(s=77, n=32, l=32, h1=4096, h2=14336, nh=32, nkvh=8, n_experts=8, topk=2, gmem=25769803776, cmem=197880360960.0, ctog_bdw=17179869184, g_bdw=306016419840, c_bdw=81604378624.0, gpu_flops=104000000000000.0, cpu_flops=1600000000000.0, tp_size=1)
status: 1
weights size: 86.5000 GB
non-experts weights size: 2.5000 GB
ctog = 0.1549 s  gpu_T = 0.1549 s  cpu_T = 0.1549 s
T = 4.957 s
gpu peak mem (decode): 20.640 GB / 21.600 GB
gpu_home: 10.43 GB
gpu_w: 10.21 GB
inter: 5.25 GB
cpu peak mem (decode): 147.282 GB / 147.432 GB
cpu_home_g: 142.32 GB
cpu_w_g: 4.96 GB
wg = 0.06  wc = 0.94  
cg = 0.00  cc = 1.00  
decode throughput = 944.66 token/s
Policy: Policy(ubs=324, n_ub=14, wg=0.055884688, wc=0.94411531, cg=0.0, cc=1.0, eg=8)
Free GPU memory: 13.1707763671875
```

For reference, on GCP g2-standard-48 instance (S2), we have
```
--gen-len 32: --ubs 324 --n-ub 14
--gen-len 64: --ubs 228 --n-ub 16
--gen-len 128: --ubs 164 --n-ub 16
--gen-len 256: --ubs 132 --n-ub 12
```

On GCP T4 (48vCPU + 192 CPU Mem) instance (S1), we have:
```
--gen-len 32: --ubs 164 --n-ub 19
--gen-len 64: --ubs 164 --n-ub 19
--gen-len 128: --ubs 164 --n-ub 14
--gen-len 256: --ubs 100 --n-ub 15
```

You may observe some variances on the generated policies. In that case, just use the generated policies to run the benchmarking.
