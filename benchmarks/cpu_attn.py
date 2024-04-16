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
        int(seq_len * (1 - random.gauss(0.5, 0.01)))
        for _ in range(BS)
    ]
seq_len_list[random.randint(0, BS - 1)] = seq_len
head_num = 32
head_dim = 128
query = torch.empty(BS, 1, head_num, head_dim, dtype=torch.float16, device="cuda")
# query = torch.empty(BS, head_num, head_dim, dtype=torch.float16)
query.uniform_(-1e-3, 1e-3)
query_cpu = query.to("cpu").to(torch.float32)
query_cpu_sdp = query_cpu.view(BS, head_num, 1, head_dim)
# query_cpu = query
query = query.view(1, BS, head_num, head_dim)

k_cache = torch.empty(BS, seq_len, head_num, head_dim, dtype=torch.float16, device="cuda")
# k_cache = torch.empty(BS, seq_len, head_num, head_dim, dtype=torch.float16)
k_cache.uniform_(-1e-3, 1e-3)
k_cache_cpu = k_cache.to("cpu").to(torch.float32)
k_cache_cpu_sdp = k_cache_cpu.transpose(1, 2)
# k_cache_cpu = k_cache
k_cache = k_cache.view(1, BS*seq_len, head_num, head_dim)

v_cache = torch.empty(BS, seq_len, head_num, head_dim, dtype=torch.float16, device="cuda")
# v_cache = torch.empty(BS, seq_len, head_num, head_dim, dtype=torch.float16)
v_cache.uniform_(-1e-3, 1e-3)
v_cache_cpu = v_cache.to("cpu").to(torch.float32)
v_cache_cpu_sdp = v_cache_cpu.transpose(1, 2)
# v_cache_cpu = v_cache
v_cache = v_cache.view(1, BS*seq_len, head_num, head_dim)

kvcache = torch.zeros((BS*seq_len, 2, int(head_num/4), head_dim), device="cuda", dtype=torch.float16)
kvcache_cpu = torch.zeros((BS*seq_len, 2, int(head_num/4), head_dim), device="cpu", dtype=torch.float16)


# multi-head attention decode stage
# q: [b, head_num, head_dim]
# k_cache: [b, seq_len, head_num, head_dim]
def multi_head_attention_decode(q, k_cache, v_cache):
    # scaling
    q = q * (head_dim**-0.5)
    # qK
    attn = torch.einsum('bhd,bshd->bhs', q, k_cache)
    # softmax
    attn = torch.softmax(attn, dim=-1)
    # qKV
    out = torch.einsum('bhs,bshd->bhd', attn, v_cache)
    return out.to(torch.float16)

# attn_mask = torch.ones(BS, head_num, 1, seq_len, dtype=torch.bool)
# for i in range(BS): 
#     attn_mask[i, :, :, seq_len_list[i]:] = 0
def torch_sdp_attention(q, k_cache, v_cache):
    out = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=None)
    return out

# reference xformer implementation
attn_op = xops.fmha.cutlass.FwOp()
attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens([1]*BS, seq_lens)
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

# compare with reference
# start = time.time()
# for i in range(10):
#     output_cpu = multi_head_attention_decode(query_cpu, k_cache_cpu, v_cache_cpu)
# print("CPU elapsed time: ", (time.time() - start)/10)


# GPU time
torch.cuda.synchronize()
start = time.time()
for i in range(10):
    # output = multi_head_attention_decode(query, k_cache, v_cache)
    output_ref = xops.memory_efficient_attention_forward(
        query,
        k_cache,
        v_cache,
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
        op=attn_op,
    ).squeeze(0)
torch.cuda.synchronize()
print("GPU elapsed time: ", (time.time() - start)/10)

# torch sdp
out = torch.zeros(BS, 1, head_num, head_dim, dtype=torch.float32, device="cpu")
seq_lens_tensor = torch.tensor(seq_lens, device="cpu")
start = time.time()
for i in range(10):
    token_attention_cpu(out, query_cpu, k_cache_cpu, v_cache_cpu, seq_lens_tensor, head_dim**-0.5)
print("cpu token attn elapsed time: ", (time.time() - start)/10)

# kv trasnfer
# torch.cuda.synchronize()
# start = time.time()
# for i in range(20):
#     kvcache.copy_(kvcache_cpu, non_blocking=True)
# torch.cuda.synchronize()
# print("kv transfer elapsed time: ", (time.time() - start)/20)


assert torch.allclose(out.to('cuda').to(torch.float16).squeeze(1), output_ref, atol=1e-5, rtol=1e-5)
print(out.squeeze(1)[:2, :2, :2])
print(output_ref[:2, :2, :2])
# output_sdp = torch_sdp_attention(query_cpu, k_cache_cpu, v_cache_cpu).squeeze(2)
# print(output_sdp[:2, :2, :2])
# print(output_ref[:2, :2, :2])
    

