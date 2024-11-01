import time
import torch

def kvcache_transfer(num_tokens, num_heads, head_dim, num_streams=2):
    kvcache = [torch.zeros((num_tokens, num_heads, head_dim), device="cuda", dtype=torch.float16) for _ in range(num_streams)]
    kvcache_cpu = [torch.zeros((num_tokens, num_heads, head_dim), device="cpu", dtype=torch.float16) for _ in range(num_streams)]
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    torch.cuda.synchronize()
    start = time.time()
    for i in range(num_streams):
        with torch.cuda.stream(streams[i]):
            kvcache[i].copy_(kvcache_cpu[i], non_blocking=True)
    torch.cuda.synchronize()
    end = time.time()
    # print(f"Time: {end - start:.5f}s")
    return end - start

# total_time = 0
# for i in range(500):
#     total_time  = total_time + kvcache_transfer(12000, 8, 128, 1)

# print(f"Average time: {total_time / 500:.5f}s")

num_tokens = 36000
num_heads = 64
head_dim = 400
comm_stream = torch.cuda.Stream()
comp_stream = torch.cuda.Stream()

kvcache = torch.zeros((num_tokens, num_heads, head_dim), device="cuda", dtype=torch.float16)
kvcache_cpu = torch.zeros((num_tokens, num_heads, head_dim), device="cpu", dtype=torch.float16)
model_cpu = torch.zeros((32, 8, 3*14336*4096), device="cpu", dtype=torch.float16)
model = torch.ones((16, 3*14336*4096), device="cuda", dtype=torch.float16)
x = torch.randn((10000, 10000), device="cuda")
y = torch.randn((10000, 10000), device="cuda")

torch.cuda.synchronize()
start = time.time()
for _ in range(1):
    index = torch.randint(0, 32, (1,)).item()
    with torch.cuda.stream(comm_stream):
        # kvcache.copy_(kvcache_cpu, non_blocking=True)
        model[:8].copy_(model_cpu[index, :], non_blocking=True)

    # with torch.cuda.stream(comp_stream):
    # z = x.matmul(y)
torch.cuda.synchronize()
end = time.time()
print(f"Time: {(end - start)/1:.5f}s")


