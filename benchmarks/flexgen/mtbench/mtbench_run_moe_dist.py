import argparse

from flexgen.flex_moe import (Policy, ExecutionEnv, CompressionConfig,
        str2bool, model_bytes, cache_bytes, hidden_bytes, get_filename)
from transformers import AutoTokenizer, AutoConfig
from flexgen.moe_pytorch_backend import (TorchDevice, TorchDisk, TorchMixedDevice, TorchTensor)
from typing import Optional
import json 
from flexgen.timer import timers
from flexgen.utils import GB 

import os
from flexgen.dist_flex_moe import add_distributed_parser_arguments, DistMixtralLM
from flexgen.dist_utils import initialize_distributed
from itertools import count
import torch
import torch.distributed as dist

def get_config(model: str, trust_remote_code: bool = True, revision: Optional[str] = None):
    config = AutoConfig.from_pretrained(
        model, trust_remote_code=trust_remote_code, revision=revision
    )
    return config

def load_questions(filename):
        questions = []
        with open(filename, "r") as fin:
            for line in fin:
                obj = json.loads(line)
                questions.append(obj)
        return questions

def comm_test(comm_device):
    # A small all_reduce for warmup.
    a = torch.ones(1).to(comm_device)
    dist.all_reduce(a)
    assert a.item() == args.world_size

def main(args):
    print(f"<run_flexgen_dist>: args.model: {args.model}")
    
    gpu_batch_size = args.gbs
    num_gpu_batches = args.num_gb
    # length_of_inputs = gpu_batch_size * num_gpu_batches
    num_inner_iterations = args.num_inner_iterations if args.num_inner_iterations is not None else args.world_size
    num_prompts = num_gpu_batches * gpu_batch_size * num_inner_iterations * 1
    print(f"Number of inner iterations: {num_inner_iterations}, world_size: {args.world_size}, number of prompts: {num_prompts}")
    
    # Prompts
    questions = load_questions("./question.jsonl")
    prompts = []
    for i in range(61):
        for question in questions:
                prompts.append(question["turns"][0])
                
    # Initialize environment
    gpu = TorchDevice(f"cuda:{args.local_rank}")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir, None, args.local_rank)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    TorchTensor.name_count = count(start=args.rank, step=args.world_size)
    
    comm_test(gpu.dev if args.comm_device == "gpu" else cpu.dev)
    
    # Tokenizer 
    print("Initialize...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # NOTE: MTBench inputs 
    all_ids_without_padding = tokenizer(prompts, padding=False).input_ids
    prompt_len = max(len(ids) for ids in all_ids_without_padding)
    print(f"Max prompt_len: {prompt_len}")
    inputs = tokenizer(prompts, padding='max_length', max_length=prompt_len)
    input_ids = inputs.input_ids 
    
    # Shrink inputs to num_prompts 
    input_ids = input_ids[:num_prompts]

    print(f"gpu_batch_sizenun: {gpu_batch_size}, num_gpu_batches: {num_gpu_batches}, len(prompts): {len(input_ids)}")
    
    policy = Policy(gpu_batch_size, num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    # NOTE: sep_layer all set to False in the policy, same as pin_weight
                    overlap=True, sep_layer=True, pin_weight=False,
                    cpu_cache_compute=args.cpu_cache_compute, attn_sparsity=1.0,
                    compress_weight=args.compress_weight,
                    comp_weight_config=CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    compress_cache=args.compress_cache,
                    comp_cache_config=CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))
    assert not (args.compress_cache), "Not implemented"
    
    mixtral_config = get_config(args.model)
    mixtral_config.pad_token_id = tokenizer.pad_token_id
    # model = DistMixtralLM(args.model, mixtral_config, env, args.path, policy)
    model = DistMixtralLM(args.model, mixtral_config, env, args.path, policy, args.rank,
                      args.world_size, args.comm_device, num_inner_iterations=num_inner_iterations,
                      async_comm=args.async_comm)
    gen_len = args.gen_len
    
    # Warmup + Actual Run 
    try:
        # print("Benchmark - Generate...")
        # timers("generate").reset()
        # output_ids = model.generate(
        #     input_ids,
        #     temperature=0,
        #     max_new_tokens=gen_len)
        # costs = timers("generate").costs
        
        print("benchmark - generate")
        for timer_name in ["generate-prompt", "generate"]:
            timers(timer_name).reset()
        output_ids = model.generate(
            input_ids, max_new_tokens=gen_len, temperature=0)
        prompt_costs = timers("generate-prompt").costs
        generate_costs = timers("generate").costs
        
    except Exception as e:
        print(f"Error: {e}")
    finally: 
        # Shutdown
        print("Shutdown...")
        env.close_copy_threads()

    if args.rank != args.world_size - 1:
        return

    # Log output
    cache_size = cache_bytes(mixtral_config, num_prompts, prompt_len + gen_len)
    hidden_size = hidden_bytes(mixtral_config, num_prompts, prompt_len + gen_len)
    
    prefill_latency = sum(prompt_costs)
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    decode_latency = sum(generate_costs)
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    log_str = (f"model size: {model_bytes(mixtral_config)/GB:.3f} GB\t"
               f"cache size: {cache_size/GB:.3f} GB\t"
               f"hidden size (prefill): {hidden_size/GB:.3f} GB\n"
               f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\n"
               f"prefill latency: {prefill_latency:.2f} s\t"
               f"prefill throughput: {prefill_throughput:.2f} token/s\n"
               f"decode latency: {decode_latency:.2f} s\t"
               f"decode throughput: {decode_throughput:.2f} token/s\n"
               f"total latency: {total_latency:.2f} s\t"
               f"total throughput: {total_throughput:.2f} token/s")
    print(log_str)
    gpu.print_stats()
    cpu.print_stats()
    
    # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # show_str = "Outputs:\n" + 70 * '-' + "\n"
    # for i in [0, len(outputs)-1]:
    #     show_str += f"{i}: {outputs[i]}\n"
    #     show_str += "-" * 70 + "\n"
    # print(show_str)
    
    if not args.no_log:
        if args.log_file == "auto":
            basename = f"rank-{args.rank}-{get_filename(args)}"
            log_filename = basename + ".log"
        else:
            log_filename = args.log_file
        with open(log_filename, "a") as fout:
            fout.write(log_str + "\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--gbs", "--gpu-batch-size", type=int)
    parser.add_argument("--num-gb", "--num-gpu-batches", type=int)
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x22B-Instruct-v0.1",
        help="The model name.")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=False)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")
    
    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    
    # Distributed arguments
    add_distributed_parser_arguments(parser)
    args = parser.parse_args()

    if args.head_ip is not None and args.port is not None:
        if args.use_mpi:
            args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
            args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
            args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
        initialize_distributed(args.head_ip, args.port, args.world_size,
                               args.rank, args.local_rank, args.comm_device)
    else:
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    assert len(args.percent) == 6

    main(args)