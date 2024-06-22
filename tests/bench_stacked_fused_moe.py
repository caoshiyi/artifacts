import torch
import time
from fastmoe.layers.stacked_fused_moe import stack_fused_moe


dtype = torch.float16
n = 14336
k = 4096
e = 8
topk = 2
w1 = torch.ones((2*e, 3 * n * k), device='cuda', dtype=dtype)
indices = torch.tensor([0,2,3,6,8,10,12,14], dtype=torch.int64, device='cuda')

for bs in [32, 64, 128, 256, 512]:
    a = torch.ones((bs, k), device='cuda', dtype=dtype)
    score = torch.randn((bs, e), device='cuda', dtype=dtype)
    triton_output = stack_fused_moe(a, w1, indices, score, topk, renormalize=True, inplace=True)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(20):
        triton_output = stack_fused_moe(a, w1, indices, score, topk, renormalize=True, inplace=True)
    torch.cuda.synchronize()
    end = time.time()
    print(f"bs {bs} time {(end - start) / 20}")
