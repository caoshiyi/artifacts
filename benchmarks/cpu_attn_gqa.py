import torch
import torch.nn.functional as F
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalFromBottomRightMask
import time
import random
import torch
from fastmoe._cpu_kernel import token_attention_cpu

seq_len = 2048
BS = 400
seq_lens = [seq_len]*BS
seq_len_list = [
        random.randint(seq_len-10, seq_len)
        for _ in range(BS)
    ]
seq_len_list[random.randint(0, BS - 1)] = seq_len
start_loc = torch.cumsum(torch.tensor([0] + seq_len_list[:-1]), 0)
head_num = 32
kv_head_num = 8
head_dim = 128
query = torch.empty(BS, 1, head_num, head_dim, dtype=torch.float16, device="cuda")
query.uniform_(-1e-3, 1e-3)
query_cpu = query.to("cpu").to(torch.float32)
query = query.view(BS, head_num, head_dim)

k_cache = torch.empty(sum(seq_len_list), kv_head_num, head_dim, dtype=torch.float16, device="cuda")
k_cache.uniform_(-1e-3, 1e-3)
k_cache_cpu = k_cache.to("cpu").to(torch.float32)


v_cache = torch.empty(sum(seq_len_list), kv_head_num, head_dim, dtype=torch.float16, device="cuda")
v_cache.uniform_(-1e-3, 1e-3)
v_cache_cpu = v_cache.to("cpu").to(torch.float32)

padded_k_cache = torch.zeros((BS, seq_len, kv_head_num, head_dim), dtype=torch.float32, device="cpu")
padded_v_cache = torch.zeros((BS, seq_len, kv_head_num, head_dim), dtype=torch.float32, device="cpu")
start = 0
for i , seq_len in enumerate(seq_len_list):
    padded_k_cache[i, :seq_len] = k_cache[start:start+seq_len].to("cpu").to(torch.float32)
    padded_v_cache[i, :seq_len] = v_cache[start:start+seq_len].to("cpu").to(torch.float32)
    start += seq_len

# reference xformer implementation
if kv_head_num != head_num:
    num_queries_per_kv = head_num // kv_head_num
    # k_cache = torch.repeat_interleave(k_cache, repeats=num_queries_per_kv, dim=1)
    # v_cache = torch.repeat_interleave(v_cache, repeats=num_queries_per_kv, dim=1)
    query = query.view(query.shape[0], kv_head_num, num_queries_per_kv,
                           query.shape[-1])
    k_cache = k_cache[:, :, None, :].expand(k_cache.shape[0], kv_head_num,
                                    num_queries_per_kv, k_cache.shape[-1])
    v_cache = v_cache[:, :,
                    None, :].expand(v_cache.shape[0], kv_head_num,
                                    num_queries_per_kv, v_cache.shape[-1])

query = query.unsqueeze(0)
k_cache = k_cache.unsqueeze(0)
v_cache = v_cache.unsqueeze(0)
attn_op = xops.fmha.cutlass.FwOp()
attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens([1]*BS, seq_len_list)
scale = float(1.0 / (head_dim**0.5))
output_ref = xops.memory_efficient_attention_forward(
    query,
    k_cache,
    v_cache,
    attn_bias=attn_bias,
    p=0.0,
    scale=scale,
    op=attn_op,
).squeeze(0)


# GPU time
torch.cuda.synchronize()
start = time.time()
for i in range(10):
    output_ref = xops.memory_efficient_attention_forward(
        query,
        k_cache,
        v_cache,
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
        op=attn_op,
    )
torch.cuda.synchronize()
print("GPU elapsed time: ", (time.time() - start)/10)

# torch sdp
out = torch.zeros(BS, 1, head_num, head_dim, dtype=torch.float32, device="cpu")
seq_lens_tensor = torch.tensor(seq_len_list, device="cpu")
start = time.time()
for i in range(10):
    token_attention_cpu(out, query_cpu, k_cache_cpu, v_cache_cpu, seq_lens_tensor, start_loc, head_dim**-0.5)
print("cpu token attn elapsed time: ", (time.time() - start)/10)

out = out.to('cuda').to(torch.float16).squeeze(1)
output_ref = output_ref.reshape(out.shape)
# print(query.squeeze(0)[:2, :2, :2])
print(out[:2, :2, :2])
print(output_ref[:2, :2, :2])
assert torch.allclose(out, output_ref, atol=1e-5, rtol=1e-5)
    

