"""Memory pool."""
import logging

import torch

logger = logging.getLogger(__name__)

class MemoryPool:
    def __init__(self, size, dtype, shape, pool_name, num_pool=1):
        self.mem_state = torch.zeros((size,), dtype=torch.int16, device="cuda")
        self.pool_name = pool_name
        self.alloc_ct = 0
        self.size = size

        self.mem_data = torch.empty((size, shape), dtype=dtype, device="cuda")

    def alloc(self, need_size):
        select_index = torch.nonzero(self.mem_state == 0).squeeze(1)[:need_size]
        if select_index.shape[0] < need_size:
            return None

        self.add_refs(select_index)
        return select_index.to(torch.int32)

    def alloc_contiguous(self, need_size):
        empty_index = torch.nonzero(self.mem_state == 0).squeeze(1)[:need_size]
        if empty_index.shape[0] < need_size:
            return None
        empty_size = len(empty_index)
        loc_sum = (
            empty_index[need_size - 1 :] - empty_index[: empty_size - (need_size - 1)]
        )
        can_used_loc = empty_index[: empty_size - (need_size - 1)][
            loc_sum == need_size - 1
        ]
        if can_used_loc.shape[0] == 0:
            return None
        start_loc = can_used_loc[0].item()
        select_index = torch.arange(start_loc, start_loc + need_size, device="cuda")
        self.add_refs(select_index)
        return select_index.to(torch.int32), start_loc, start_loc + need_size

    def free(self, free_index):
        return self.decrease_refs(free_index)

    def used_size(self):
        return len(torch.nonzero(self.mem_state).squeeze(1))

    def available_size(self):
        return torch.sum(self.mem_state == 0).item()

    def add_refs(self, token_index: torch.Tensor):
        self.alloc_ct += len(token_index)
        self.mem_state[token_index] += 1

    def decrease_refs(self, token_index: torch.Tensor):
        self.alloc_ct -= len(token_index)
        self.mem_state[token_index] -= 1

        num_freed = torch.sum(self.mem_state[token_index] == 0)
        return num_freed

    def clear(self):
        self.mem_state.fill_(0)
        self.alloc_ct = 0


class ReqToTokenPool:
    def __init__(self, size, max_context_len, num_layers, device="cuda"):
        self.mem_state = torch.ones((size,), dtype=torch.bool, device="cuda")
        self.can_use_mem_size = size
        if device == "cuda":
            self.req_to_token = torch.empty(
                (size, num_layers, max_context_len), dtype=torch.int32, device="cuda"
            )
        else:
            self.req_to_token = torch.empty(
                (size, max_context_len), dtype=torch.int32, device="cpu"
            )

    def alloc(self, need_size):
        if need_size > self.can_use_mem_size:
            return None

        select_index = torch.nonzero(self.mem_state).squeeze(1)[:need_size]
        self.mem_state[select_index] = 0
        self.can_use_mem_size -= need_size
        return select_index.to(torch.int32)

    def free(self, free_index):
        if isinstance(free_index, (int,)):
            self.can_use_mem_size += 1
        else:
            self.can_use_mem_size += free_index.shape[0]
        self.mem_state[free_index] = 1

        # if self.can_use_mem_size == len(self.mem_state):
        #     print(f"ReqToTokenPool: freed all. size = {self.can_use_mem_size}.")

    def clear(self):
        self.mem_state.fill_(1)
        self.can_use_mem_size = len(self.mem_state)


class TokenToKVPool:
    def __init__(self, gpu_size, cpu_size, dtype, head_num, head_dim, layer_num):
        self.gpu_mem_state = torch.zeros((gpu_size,), dtype=torch.int16, device="cuda")
        self.cpu_mem_state = torch.zeros((cpu_size,), dtype=torch.int16, device="cpu")
        self.alloc_ct = 0
        self.alloc_ct_cpu = 0

        # [size, key/value, head_num, head_dim] for each layer
        self.kv_data = torch.empty((gpu_size, 2, head_num, head_dim), dtype=dtype, device="cuda")
        

        self.kv_data_cpu = [
            torch.empty((cpu_size, 2, head_num, head_dim), dtype=dtype, device="cpu")
            for _ in range(layer_num)
        ]

    # gpu_indices: [num_layer, num_token], cpu_indices: [num_token]
    def offload_kv_cache(self, gpu_indices, cpu_indices=None, other_gpu_indices=None, start_layer=0):
        if other_gpu_indices is not None:
            self.decrease_refs(other_gpu_indices.flatten())
        if cpu_indices is None:
            need_size = gpu_indices.shape[1] # num_token
            select_index = torch.nonzero(self.cpu_mem_state == 0).squeeze(1)[:need_size]
            assert select_index.shape[0] >= need_size
            self.cpu_mem_state[select_index] += 1
            self.alloc_ct_cpu += need_size
            self.decrease_refs(gpu_indices.flatten())
            # todo @caoshiyi optimize, use kernels
            for layer in range(gpu_indices.shape[0]):
                for i, index in enumerate(select_index):
                    self.kv_data_cpu[layer][index, :].copy_(self.kv_data[gpu_indices[layer, i], :])
            return select_index
        else:
            self.decrease_refs(gpu_indices.flatten())
            # todo @caoshiyi optimize, use kernels
            for layer in range(gpu_indices.shape[0]):
                for i, index in enumerate(cpu_indices):
                    self.kv_data_cpu[start_layer+layer][index, :].copy_(self.kv_data[gpu_indices[layer, i], :])
            return cpu_indices


    def load_kv_cache(self, cpu_indices, layer_ids):
        need_size = len(cpu_indices) * len(layer_ids) #num_token * num_layer
        select_index = torch.nonzero(self.gpu_mem_state == 0).squeeze(1)[:need_size]
        assert select_index.shape[0] >= need_size
        select_index = select_index.view(len(layer_ids), len(cpu_indices))
        # todo @caoshiyi optimize, use kernels
        for i, layer in enumerate(layer_ids):
            for j, index in enumerate(cpu_indices):
                self.kv_data[select_index[i, j], :].copy_(self.kv_data_cpu[layer][index, :])
        self.add_refs(select_index)
        return select_index.to(torch.int32)

    def get_key_buffer(self):
        return self.kv_data[:, 0]

    def get_value_buffer(self):
        return self.kv_data[:, 1]

    def alloc(self, need_size):
        select_index = torch.nonzero(self.gpu_mem_state == 0).squeeze(1)[:need_size]
        if select_index.shape[0] < need_size:
            return None

        self.add_refs(select_index)
        return select_index.to(torch.int32)

    # todo, alloc_multi_contiguous
    def alloc_contiguous(self, need_size):
        empty_index = torch.nonzero(self.gpu_mem_state == 0).squeeze(1)[:need_size]
        if empty_index.shape[0] < need_size:
            return None
        empty_size = len(empty_index)
        loc_sum = (
            empty_index[need_size - 1 :] - empty_index[: empty_size - (need_size - 1)]
        )
        can_used_loc = empty_index[: empty_size - (need_size - 1)][
            loc_sum == need_size - 1
        ]
        if can_used_loc.shape[0] == 0:
            return None

        start_loc = can_used_loc[0].item()
        select_index = torch.arange(start_loc, start_loc + need_size, device="cuda")
        self.add_refs(select_index)
        return select_index.to(torch.int32), start_loc, start_loc + need_size

    def free(self, free_index):
        return self.decrease_refs(free_index)

    def used_size(self):
        return len(torch.nonzero(self.gpu_mem_state).squeeze(1))

    def available_size(self):
        return torch.sum(self.gpu_mem_state == 0).item()
    
    def cpu_available_size(self):
        return torch.sum(self.cpu_mem_state == 0).item()

    def add_refs(self, token_index: torch.Tensor):
        self.alloc_ct += len(token_index)
        self.gpu_mem_state[token_index] += 1

    def decrease_refs(self, token_index: torch.Tensor):
        self.alloc_ct -= len(token_index)
        self.gpu_mem_state[token_index] -= 1

        num_freed = torch.sum(self.gpu_mem_state[token_index] == 0)

        # if self.alloc_ct == 0:
        #     print(f"TokenToKVPool: freed all. size = {len(self.mem_state)}.")

        return num_freed

    def clear(self):
        self.gpu_mem_state.fill_(0)
        self.cpu_mem_state.fill_(0)
        self.alloc_ct = 0
        self.alloc_ct_cpu = 0
