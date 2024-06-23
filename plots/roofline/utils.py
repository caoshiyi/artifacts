import numpy as np

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

def attention_flops(batch_size, kv_sequence_length, num_q_head, head_dim):
    # FLOPs for dot products between 1 query and all keys (each key has 'head_dim' elements)
    dot_product_flops = batch_size * 1 * kv_sequence_length * num_q_head * head_dim * 2
    # Softmax and summing over all kv pairs for each query head
    softmax_sum_flops = batch_size * 1 * kv_sequence_length * num_q_head * 2
    return dot_product_flops + softmax_sum_flops

def attention_bytes(batch_size, kv_sequence_length, num_q_head, num_kv_head, head_dim, dtype=np.float16):
    # Memory for 1 query, all keys, and all values
    type_size = dtype_size(dtype)
    query_memory = batch_size * 1 * num_q_head * head_dim * type_size
    key_memory = batch_size * kv_sequence_length * num_kv_head * head_dim * type_size
    value_memory = batch_size * kv_sequence_length * num_kv_head * head_dim * type_size
    # Output memory for the results of the attention mechanism
    output_memory = batch_size * 1 * num_q_head * head_dim * type_size
    return query_memory + key_memory + value_memory + output_memory

def MLP_flops(hidden_size1, hidden_size2, batch_size, topk):
    return 2 * batch_size * hidden_size1 * hidden_size2 * 3 * topk

def MLP_bytes(hidden_size1, hidden_size2, batch_size, n_experts):
    # Input, output, and weights; biases are typically small and can be neglected for this estimation
    input_bytes = 2 * batch_size * hidden_size1  # Input data bytes
    output_bytes = 2 * batch_size * hidden_size2  # Output data bytes
    weight_bytes = 2 * hidden_size1 * hidden_size2 * 3 * n_experts  # Weight data bytes
    return input_bytes + output_bytes + weight_bytes
