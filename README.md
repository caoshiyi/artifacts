# MoE-Lightning: Artifact Evaluation

This document contains instruction for ASPLOS 2025 artifact evaluation for the paper *MoE-Lightning: High-Throughput MoE Inference on
Memory-constrained GPUs*. 

## Artifact Overview 

## Installation 
```
git clone https://github.com/caoshiyi/FastMoE.git
cd FastMoE
conda create -n fastmoe python=3.11
conda activate fastmoe
conda install onemkl-sycl-blas
conda install mkl-include
pip install -e .
pip install triton==2.2.0
```

## Run tests
The first-time weights loading can take ~10min.
```
python -m fastmoe.serve.launch_server --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 --port 30000 --cpu-mem-bdw 100 --avg-prompt-len 77 --gen-len 32
cd benchmarks/mtbench
python bench.py --port 30000
```