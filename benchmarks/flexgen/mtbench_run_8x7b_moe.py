"""Complete sentences with FlexGen and OPT models."""
import argparse

from flexgen.flex_moe import (Policy, MixtralLM, ExecutionEnv, CompressionConfig,
        str2bool, model_bytes, cache_bytes, hidden_bytes)
from transformers import AutoTokenizer, AutoConfig
from flexgen.moe_pytorch_backend import (TorchDevice, TorchDisk, TorchMixedDevice)
from typing import Optional
import json 
from flexgen.timer import timers
from flexgen.utils import GB 

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
    
def main(args):
    gpu_batch_size = args.gbs
    num_gpu_batches = args.num_gb
    length_of_inputs = gpu_batch_size * num_gpu_batches
    
    # Prompts
    questions = load_questions("../mtbench/question.jsonl")
    prompts = []
    for i in range(61):
        for question in questions:
                prompts.append(question["turns"][0])
                
    # Initialize environment
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    
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
    print(f"Number of prompts: {len(prompts)}, number of input ids: {len(input_ids)}")
    
    # Shrink inputs to length_of_inputs
    input_ids = input_ids[:length_of_inputs]

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
    model = MixtralLM(args.model, mixtral_config, env, args.path, policy)
    gen_len = args.gen_len
    
    # Warmup + Actual Run 
    try:
        # print("Warmup...")
        # warmup_inputs = tokenizer(prompts, padding='max_length', max_length=prompt_len)
        # output_ids = model.generate(
        #     warmup_inputs.input_ids,
        #     temperature=0,
        #     max_new_tokens=2)
        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        print("Benchmark - Generate...")
        timers("generate").reset()
        output_ids = model.generate(
            input_ids,
            temperature=0,
            max_new_tokens=gen_len)
        costs = timers("generate").costs
        
        # Log output 
        print("logging...")
        
        num_prompts = len(input_ids)
        # TODO: propmt length takes the max padding length 
        cache_size = cache_bytes(mixtral_config, num_prompts, prompt_len + gen_len)
        hidden_size = hidden_bytes(mixtral_config, num_prompts, prompt_len + gen_len)
        
        prefill_latency = costs[0]
        prefill_throughput = num_prompts * prompt_len / prefill_latency
        decode_latency = sum(costs[1:])
        decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
        num_generated_tokens = num_prompts * gen_len
        total_latency = prefill_latency + decode_latency
        total_throughput = num_generated_tokens / total_latency
        _, gpu_peak_mem = gpu.mem_stats()
        _, cpu_peak_mem = cpu.mem_stats()
    
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"

        gpu.print_stats()
        cpu.print_stats()
        log_str = (f"model size: {model_bytes(mixtral_config)/GB:.3f} GB\t"
               f"cache size: {cache_size/GB:.3f} GB\t"
               f"hidden size (p): {hidden_size/GB:.3f} GB\n"
               f"peak gpu mem: {gpu_peak_mem / GB:.3f} GB\t"
               f"prefill latency: {prefill_latency:.3f} s\t"
               f"prefill throughput: {prefill_throughput:.3f} token/s\n"
               f"decode latency: {decode_latency:.3f} s\t"
               f"decode throughput: {decode_throughput:.3f} token/s\n"
               f"total latency: {total_latency:.3f} s\t"
               f"total throughput: {total_throughput:.3f} token/s")
        
        print(log_str)
    except Exception as e:
        print(f"Error: {e}")
    finally: 
        # Shutdown
        print("Shutdown...")
        env.close_copy_threads()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--gbs", "--gpu-batch-size", type=int)
    parser.add_argument("--num-gb", "--num-gpu-batches", type=int)
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1",
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
    
    # TODO: originally False but all results are True 
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=False)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")
    args = parser.parse_args()

    assert len(args.percent) == 6

    main(args)