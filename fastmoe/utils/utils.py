import random
import socket
import sys
import time
import traceback
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist

def get_available_gpu_memory(gpu_id, distributed=True):
    """
    Get available memory for cuda:gpu_id device.
    When distributed is True, the available memory is the minimum available memory of all GPUs.
    """
    import torch

    num_gpus = torch.cuda.device_count()
    assert gpu_id < num_gpus

    if torch.cuda.current_device() != gpu_id:
        print(
            f"WARN: current device is not {gpu_id}, but {torch.cuda.current_device()}, ",
            "which may cause useless memory allocation for torch CUDA context.",
        )

    free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)

    if distributed:
        tensor = torch.tensor(free_gpu_memory, dtype=torch.float32).to(
            torch.device("cuda", gpu_id)
        )
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        free_gpu_memory = tensor.item()

    return free_gpu_memory / (1 << 30)


def set_random_seed(seed: int) -> None:
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def alloc_usable_network_port(num, used_list=()):
    port_list = []
    for port in range(10000, 65536):
        if port in used_list:
            continue

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                port_list.append(port)
            except socket.error:
                pass

            if len(port_list) == num:
                return port_list
    return None


def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except socket.error:
            return False


def handle_port_init(
    port: Optional[int] = None,
    additional_ports: Optional[List[int]] = None,
    tp_size: int = 1,
):
    port = 30000 if port is None else port
    additional_ports = [] if additional_ports is None else additional_ports
    additional_ports = (
        [additional_ports] if isinstance(additional_ports, int) else additional_ports
    )
    # first check on server port
    if not check_port(port):
        new_port = alloc_usable_network_port(1, used_list=[port])[0]
        print(f"Port {port} is not available, using {new_port} instead.")
        port = new_port

    # then we check on additional ports
    additional_unique_ports = set(additional_ports) - {port}
    # filter out ports that are already in use
    can_use_ports = [port for port in additional_unique_ports if check_port(port)]

    num_specified_ports = len(can_use_ports)
    if num_specified_ports < 4 + tp_size:
        addtional_can_use_ports = alloc_usable_network_port(
            num=4 + tp_size - num_specified_ports, used_list=can_use_ports + [port]
        )
        can_use_ports.extend(addtional_can_use_ports)

    additional_ports = can_use_ports[: 4 + tp_size]
    return port, additional_ports


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def get_int_token_logit_bias(tokenizer, vocab_size):
    from transformers import LlamaTokenizer, LlamaTokenizerFast

    # a bug when model's vocab size > tokenizer.vocab_size
    vocab_size = tokenizer.vocab_size
    logit_bias = np.zeros(vocab_size, dtype=np.float32)
    for t_id in range(vocab_size):
        ss = tokenizer.decode([t_id]).strip()
        if not (ss.isdigit() or len(ss) == 0 or t_id == tokenizer.eos_token_id):
            logit_bias[t_id] = -1e5
        # else:
        #    print(ss, t_id)

    return logit_bias


def wrap_kernel_launcher(kernel):
    """A faster launcher for triton kernels."""
    import torch.distributed as dist

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    kernels = kernel.cache[rank].values()
    kernel = next(iter(kernels))

    # Different trition versions use different low-level names
    if hasattr(kernel, "cu_function"):
        kfunction = kernel.cu_function
    else:
        kfunction = kernel.function

    if hasattr(kernel, "c_wrapper"):
        run = kernel.c_wrapper
    else:
        run = kernel.run

    add_cluster_dim = True

    def ret_func(grid, num_warps, *args):
        nonlocal add_cluster_dim

        try:
            if add_cluster_dim:
                run(
                    grid[0],
                    grid[1],
                    grid[2],
                    num_warps,
                    1,
                    1,
                    1,
                    1,
                    kernel.shared,
                    0,
                    kfunction,
                    None,
                    None,
                    kernel,
                    *args,
                )
            else:
                run(
                    grid[0],
                    grid[1],
                    grid[2],
                    num_warps,
                    kernel.shared,
                    0,
                    kfunction,
                    None,
                    None,
                    kernel,
                    *args,
                )
        except TypeError:
            add_cluster_dim = not add_cluster_dim
            ret_func(grid, num_warps, *args)

    return ret_func