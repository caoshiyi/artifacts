import torch
from fastmoe._cpu_kernel import token_attention_cpu
from fastmoe.layers.context_flashattention_nopad import context_attention_fwd
from fastmoe.layers.token_attention import token_attention_fwd
from fastmoe.serve.router.model_runner import ForwardMode, InputMetadata, DecodePart
from torch import nn


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scaling,
        num_kv_heads,
        layer_id,
    ):
        super().__init__()

        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.layer_id = layer_id

        self.prefill_forward = self.prefill_forward_triton
        self.decode_forward = self.decode_forward_triton
        self.partial_forward = self.partial_forward_triton
        self.partial_decode_forward = self.partial_decode_forward_triton
        self.cpu_decode_forward = self.cpu_decode_forward_cpp

    def prefill_forward_triton(self, q, k, v, input_metadata: InputMetadata):
        o = torch.empty_like(q)

        context_attention_fwd(
            q.view(-1, self.tp_q_head_num, self.head_dim),
            k,
            v,
            o.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.start_loc,
            input_metadata.seq_lens,
            input_metadata.max_seq_len,
        )
        self.store_kv_cache(k, v, input_metadata)

        return o
    
    def partial_forward_triton(self, q, k, v, input_metadata: InputMetadata):
        o = torch.empty_like(q)

        context_attention_fwd(
            q.view(-1, self.tp_q_head_num, self.head_dim),
            k,
            v,
            o.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.start_loc,
            input_metadata.seq_lens,
            input_metadata.max_seq_len,
        )
        self.store_kv_cache_partial(k, v, input_metadata)

        return o

    def decode_forward_triton(self, q, k, v, input_metadata: InputMetadata):
        o = torch.empty_like(q)
        self.store_kv_cache(k, v, input_metadata)

        token_attention_fwd(
            q.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id),
            input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id),
            o.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.req_to_token_pool.req_to_token,
            input_metadata.req_pool_indices,
            input_metadata.start_loc,
            input_metadata.seq_lens,
            input_metadata.max_seq_len,
            input_metadata.other_kv_index,
            input_metadata.total_num_tokens,
        )

        return o
    
    def partial_decode_forward_triton(self, q, k, v, input_metadata: InputMetadata):
        o = torch.empty_like(q)
        self.store_kv_cache_partial(k, v, input_metadata)

        token_attention_fwd(
            q.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.token_to_kv_pool.get_key_buffer(),
            input_metadata.token_to_kv_pool.get_value_buffer(),
            o.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.req_to_token_pool.req_to_token,
            input_metadata.req_pool_indices,
            input_metadata.start_loc,
            input_metadata.seq_lens,
            input_metadata.max_seq_len,
            input_metadata.other_kv_index.item(),
            input_metadata.total_num_tokens,
        )

        return o

    def cpu_decode_forward_cpp(self, input_metadata: InputMetadata):
        qkv = input_metadata.qkv_pin.view(-1, self.tp_q_head_num + 2 * self.tp_k_head_num, self.head_dim)
        query = qkv[:, :self.tp_q_head_num, :].view(-1, 1, self.tp_q_head_num, self.head_dim)
        k = qkv[:, self.tp_q_head_num : self.tp_q_head_num + self.tp_k_head_num, :].view(-1, self.tp_k_head_num, self.head_dim)
        v = qkv[:, self.tp_q_head_num + self.tp_k_head_num :, :].view(-1, self.tp_v_head_num, self.head_dim)

        self.store_kv_cache_cpu(k, v, input_metadata)
        
        print("query:", query[:4, 0, :2, :2], "layer:", self.layer_id)
        token_attention_cpu(
            input_metadata.hidden_pin, query, 
            input_metadata.token_to_kv_pool.get_key_buffer_cpu(self.layer_id), 
            input_metadata.token_to_kv_pool.get_value_buffer_cpu(self.layer_id), 
            input_metadata.seq_lens, 
            input_metadata.start_loc, 
            self.head_dim**-0.5
        )
        print("hidden_pin:", input_metadata.hidden_pin[:4, 0, :2, :2])

        return
    
    def forward(self, q, k, v, input_metadata: InputMetadata):
        if k is not None and v is not None:
            k = k.view(-1, self.tp_k_head_num, self.head_dim)
            v = v.view(-1, self.tp_v_head_num, self.head_dim)

        if input_metadata.forward_mode == ForwardMode.PREFILL:
            return self.prefill_forward(q, k, v, input_metadata)
        elif input_metadata.forward_mode == ForwardMode.DECODE:
            return self.decode_forward(q, k, v, input_metadata)
        elif input_metadata.forward_mode == ForwardMode.PARTIAL:
            return self.partial_forward(q, k, v, input_metadata)
        elif input_metadata.forward_mode == ForwardMode.PARTIAL_DECODE:
            if input_metadata.decode_part == DecodePart.CPU_ATTN:
                return self.cpu_decode_forward_cpp(input_metadata)
            else:
                return self.partial_decode_forward(q, k, v, input_metadata)

    def store_kv_cache(self, cache_k, cache_v, input_metadata: InputMetadata):
        key_buffer = input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id)
        value_buffer = input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id)
        if input_metadata.out_cache_loc is not None:
            key_buffer[input_metadata.out_cache_loc] = cache_k
            value_buffer[input_metadata.out_cache_loc] = cache_v
        elif input_metadata.out_cache_cont_start is not None:
            key_buffer[
                input_metadata.out_cache_cont_start : input_metadata.out_cache_cont_end
            ] = cache_k
            value_buffer[
                input_metadata.out_cache_cont_start : input_metadata.out_cache_cont_end
            ] = cache_v
        else:
            raise RuntimeError()
        
    def store_kv_cache_cpu(self, cache_k, cache_v, input_metadata: InputMetadata):
        key_buffer = input_metadata.token_to_kv_pool.get_key_buffer_cpu(self.layer_id)
        value_buffer = input_metadata.token_to_kv_pool.get_value_buffer_cpu(self.layer_id)
        key_buffer[input_metadata.out_cache_loc] = cache_k
        value_buffer[input_metadata.out_cache_loc] = cache_v
    
    def store_kv_cache_partial(self, cache_k, cache_v, input_metadata: InputMetadata):
        key_buffer = input_metadata.token_to_kv_pool.get_key_buffer()
        value_buffer = input_metadata.token_to_kv_pool.get_value_buffer()
        if input_metadata.out_cache_loc is not None:
            key_buffer[input_metadata.out_cache_loc] = cache_k
            value_buffer[input_metadata.out_cache_loc] = cache_v
        else:
            raise RuntimeError()
