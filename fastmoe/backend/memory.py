"""Memory pool."""
import logging

import torch

logger = logging.getLogger(__name__)

class TokenToKVPool:
    def __init__(self, gpu_size, cpu_size, dtype, head_num, head_dim, layer_num):
        self.cur_start_loc = 0
        self.cur_cl = 0
        self.dtype = dtype
        self.cache_line = gpu_size // 2
        self.head_num = head_num
        self.head_dim = head_dim

        # round cpu_size to be a multiple of (gpu_size // 2)
        self.cpu_size = (cpu_size // self.cache_line) * self.cache_line

        # [size, key/value, head_num, head_dim]
        # This serves as a direct mapped cache on gpu for prefill stage
        self.kv_data = torch.empty((2 * self.cache_line, 2, head_num, head_dim), dtype=dtype, device="cuda")
        

        self.kv_data_cpu = [
            torch.empty((self.cpu_size, 2, head_num, head_dim), dtype=dtype, device="cpu")
            for _ in range(layer_num)
        ]
    
    def store(self, dst_start, size, layer):
        src_start = dst_start % (2 * self.cache_line)
        dst_end = dst_start + size
        src_end = src_start + size
        self.kv_data_cpu[layer][dst_start:dst_end, :, :, :].copy_(self.kv_data[src_start:src_end, :, :, :], non_blocking=True)
    
    def load(self, src_start, size, layer):
        dst_start = src_start % (2 * self.cache_line)
        src_end = src_start + size
        dst_end = dst_start + size
        self.kv_data[dst_start:dst_end, :, :, :].copy_(self.kv_data_cpu[layer][src_start:src_end, :, :, :], non_blocking=True)

    def get_key_buffer(self):
        return self.kv_data[:, 0, :, :]

    def get_value_buffer(self):
        return self.kv_data[:, 1, :, :]
    
    def get_key_buffer_cpu(self, layer):
        return self.kv_data_cpu[layer][:, 0, :, :]
    
    def get_value_buffer_cpu(self, layer):
        return self.kv_data_cpu[layer][:, 1, :, :]
    
    def alloc_cpu(self):
        start_loc = self.cur_start_loc
        self.cur_start_loc = self.cur_start_loc + self.cache_line
        cl_idx = self.cur_cl
        self.cur_cl = (self.cur_cl + 1) % 2
        return start_loc, cl_idx * self.cache_line, cl_idx

    def delete_gpu_cache(self):
        self.kv_data = None

    def clear(self):
        self.cur_start_loc = 0
        self.cur_cl = 0
