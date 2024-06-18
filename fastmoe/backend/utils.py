import dataclasses
from fastmoe.utils.model_config import ModelConfig
import numpy as np

KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
T = 1e12

@dataclasses.dataclass
class Policy:
    ubs: int
    n_ub: int

    wg: float
    wc: float
    cg: float
    cc: float

    eg: int

@dataclasses.dataclass
class HardwareConfig:
    gmem: int = 24 * GB
    cmem: int = 192 * GB

    ctog_bdw: float = 16 * GB
    g_bdw: float = 300 * GB
    c_bdw: float = 100 * GB

    gpu_flops: float = 104 * T
    cpu_flops: float = 13 * T

    tp_size: int = 1

    @classmethod
    def init(cls, gpu_device_name, cpu_mem, c_bdw, tp_size):
        if "L4" in gpu_device_name:
            return cls(
                gmem=24 * GB,
                cmem=cpu_mem * GB,
                ctog_bdw=16 * GB,
                g_bdw=285 * GB,
                c_bdw=c_bdw * GB,
                gpu_flops=104 * T,
                cpu_flops=1.6 * T,
                tp_size=tp_size
            )
        elif "T4" in gpu_device_name:
            return cls(
                gmem=15 * GB,
                cmem=cpu_mem * GB,
                ctog_bdw=16 * GB,
                g_bdw=300 * GB,
                c_bdw=c_bdw * GB,
                gpu_flops=65 * T,
                cpu_flops=0.8 * T,
                tp_size=tp_size
            )
        else:
            return cls

@dataclasses.dataclass
class CostModelConfig:
    s: int = 512
    n: int = 32

    l: int = 32
    h1: int = 4096
    h2: int = 4096 * 4
    nh: int = 32
    nkvh: int = 8

    n_experts: int = 8
    topk: int = 2

    gmem: int = 24 * GB
    cmem: int = 192 * GB

    ctog_bdw: float = 16 * GB
    g_bdw: float = 300 * GB
    c_bdw: float = 100 * GB

    gpu_flops: float = 104 * T
    cpu_flops: float = 13 * T

    tp_size: int = 1

    @classmethod
    def init(cls, model_config: ModelConfig, hardware_config: HardwareConfig, prompt_len, gen_len):
        tp_size = hardware_config.tp_size
        return cls(
            s=prompt_len,
            n=gen_len,
            l=model_config.num_hidden_layers,
            h1=model_config.hidden_size,
            h2=model_config.intermediate_size,
            nh=model_config.num_attention_heads,
            nkvh=model_config.num_key_value_heads,
            n_experts=model_config.num_local_experts,
            topk=model_config.topk,
            gmem=hardware_config.gmem * tp_size,
            cmem=hardware_config.cmem,
            ctog_bdw=hardware_config.ctog_bdw,
            g_bdw=hardware_config.g_bdw * tp_size,
            c_bdw=hardware_config.c_bdw,
            gpu_flops=hardware_config.gpu_flops * tp_size,
            cpu_flops=hardware_config.cpu_flops,
            tp_size=tp_size
        )
    
def dtype_size(dtype):
    if dtype == 'f32':
        return 4
    elif dtype == 'f16':
        return 2
    elif dtype == 'int8':
        return 1
    elif dtype == 'int4':
        return 0.5
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def attention_flops(batch_size, query_length, kv_sequence_length, num_q_head, head_dim):
    # FLOPs for dot products between 1 query and all keys (each key has 'head_dim' elements)
    dot_product_flops = batch_size * query_length * kv_sequence_length * num_q_head * head_dim * 2
    # Softmax and summing over all kv pairs for each query head
    softmax_sum_flops = batch_size * query_length * kv_sequence_length * num_q_head * 2
    return dot_product_flops + softmax_sum_flops

def attention_bytes(batch_size, query_length, kv_sequence_length, num_q_head, num_kv_head, head_dim, dtype="f16"):
    # Memory for 1 query, all keys, and all values
    type_size = dtype_size(dtype)
    query_memory = batch_size * query_length * num_q_head * head_dim * type_size
    key_memory = batch_size * kv_sequence_length * num_kv_head * head_dim * type_size
    value_memory = batch_size * kv_sequence_length * num_kv_head * head_dim * type_size
    # Output memory for the results of the attention mechanism
    output_memory = batch_size * query_length * num_q_head * head_dim * type_size
    return query_memory + key_memory + value_memory + output_memory


def MLP_flops(hidden_size1, hidden_size2, batch_size, topk):
    return 2 * batch_size * hidden_size1 * hidden_size2 * 3 * topk

def MLP_bytes(hidden_size1, hidden_size2, batch_size, n_experts):
    # Input, output, and weights; biases are typically small and can be neglected for this estimation
    input_bytes = 2 * batch_size * hidden_size1  # Input data bytes
    output_bytes = 2 * batch_size * hidden_size2  # Output data bytes
    weight_bytes = 2 * hidden_size1 * hidden_size2 * 3 * n_experts  # Weight data bytes
    return input_bytes + output_bytes + weight_bytes

