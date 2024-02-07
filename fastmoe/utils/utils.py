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