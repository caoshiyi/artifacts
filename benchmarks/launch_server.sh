python -m fastmoe.serve.launch_server --model-path huggyllama/llama-7b --port 30000

CUDA_VISIBLE_DEVICES=0 python -m fastmoe.serve.launch_server --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 --port 30000