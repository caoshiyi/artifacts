import argparse
import numpy as np
import pulp
from fastmoe.backend.utils import (MLP_flops, MLP_bytes, 
                                   attention_flops, attention_bytes, 
                                   Policy, HardwareConfig, CostModelConfig)
from fastmoe.utils.model_config import ModelConfig

KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
T = 1e12

def solve_lp(config, bls, gbs, verbose=1, stage="decode"):
    assert bls > 0 and gbs > 0
    assert bls >= gbs and bls % gbs == 0
    gattn = 0

    ## Constants
    s = config.s
    n = config.n
    l = config.l
    h1 = config.h1
    h2 = config.h2
    nh = config.nh
    nkvh = config.nkvh
    ne = config.n_experts
    topk = config.topk

    gmem = config.gmem
    cmem = config.cmem

    ctog_bdw = config.ctog_bdw
    g_bdw = config.g_bdw
    c_bdw = config.c_bdw

    gpu_flops = config.gpu_flops
    cpu_flops = config.cpu_flops

    ## Create Problem
    prob = pulp.LpProblem('storage', sense=pulp.LpMinimize)

    ## Create variables for cost
    T = pulp.LpVariable("T", lowBound=0)
    ctog = pulp.LpVariable("ctog_i", lowBound=0)
    cpu_comp = pulp.LpVariable("comp_i^c", lowBound=0)
    cpu_comm = pulp.LpVariable("comm_i^c", lowBound=0)
    gpu_comp_attn = pulp.LpVariable("attn_comp_i^g", lowBound=0)
    gpu_comm_attn = pulp.LpVariable("attn_comm_i^g", lowBound=0)
    gpu_T_attn = pulp.LpVariable("gpu_t_attn", lowBound=0)
    gpu_comp_mlp = pulp.LpVariable("mlp_comp_i^g", lowBound=0)
    gpu_comm_mlp = pulp.LpVariable("mlp_comm_i^g", lowBound=0)
    gpu_T_mlp = pulp.LpVariable("gpu_t_mlp", lowBound=0)
    gpu_T = pulp.LpVariable("gpu_t", lowBound=0)
    cpu_T = pulp.LpVariable("cpu_t", lowBound=0)

    wg = pulp.LpVariable("wg", lowBound=0)
    wc = pulp.LpVariable("wc", lowBound=0)
    cg = pulp.LpVariable("cg", lowBound=0)
    cc = pulp.LpVariable("cc", lowBound=0)

    ## Set objective

    # Minimize T/bls
    prob += T * (1 / bls)

    # layer weight size (QOProj, KVProj, Experts)
    num_kv_group = nh // nkvh
    h_kv = h1 // num_kv_group
    w_other = (4 * h1 ** 2 + 4 * h1 ** 2 // num_kv_group) * l
    wi = 2 * 3 * h1 * h2 * ne

    # --------------- Add constraints -------------------

    prob += wg + wc == 1
    if gattn == 1:
        prob += cg + cc == 1
    else:
        prob += cg == 0
        prob += cc == 1

    # T = max(ctogp, gtocp, cpu_T, gpu_T)
    prob += T >= ctog
    prob += T >= cpu_T
    prob += T >= gpu_T

    if stage == "decode":
        # ctogp = (weight_ctog + cache_ctog) / ctog_bdw
        prob += ctog == (1 / ctog_bdw) * (wi * wc
                        + 4 * (s + n / 2) * h_kv * bls * cc * gattn)
    elif stage == "prefill":
        prob += ctog == (1 / ctog_bdw) * (wi * wc + 2 * bls * s * h1)
        # prob += gtoc == (1 / ctog_bdw) * (4 * s * h_kv * bls * cg)

    
    # gpu_t = gpu_T_attn + gpu_T_mlp
    prob += gpu_T == gpu_T_attn + gpu_T_mlp
    # gpu_T_mlp = max(gpu_comp_mlp, gpu_comm_mlp)
    prob += gpu_T_mlp >= gpu_comp_mlp
    prob += gpu_T_mlp >= gpu_comm_mlp
    if stage == "decode":
        prob += gpu_comp_mlp == (1 / gpu_flops) * MLP_flops(h1, h2, gbs, topk) * (bls // gbs)
        prob += gpu_comm_mlp == (1 / g_bdw) * MLP_bytes(h1, h2, gbs, ne) * (bls // gbs)
    elif stage == "prefill":
        prob += gpu_comp_mlp == (1 / gpu_flops) * (MLP_flops(h1, h2, gbs*s, topk)) * (bls // gbs)
        prob += gpu_comm_mlp == (1 / g_bdw) * (MLP_bytes(h1, h2, gbs*s, ne)) * (bls // gbs)

    # gpu_T_attn = max(gpu_comp_attn, gpu_comm_attn)
    prob += gpu_T_attn >= gpu_comp_attn
    prob += gpu_T_attn >= gpu_comm_attn
    if stage == "decode":
        prob += gpu_comp_attn == (1 / gpu_flops) * attention_flops(gbs, 1, s + n, nh, h1 / nh) * (bls // gbs) * gattn
        prob += gpu_comm_attn == (1 / g_bdw) * attention_bytes(gbs, 1, s + n, nh, nkvh, h1 / nh) * (bls // gbs) * gattn
    elif stage == "prefill":
        prob += gpu_comp_attn == (1 / gpu_flops) * (attention_flops(gbs, s, s, nh, h1 / nh)) * (bls // gbs)
        prob += gpu_comm_attn == (1 / g_bdw) * (attention_bytes(gbs, s, s, nh, nkvh, h1 / nh)) * (bls // gbs)
    
    
    # cpu_t = max(cpu_comp, cpu_comm)
    if stage == "decode":
        prob += cpu_T >= cpu_comp
        prob += cpu_T >= cpu_comm
        
        prob += cpu_comp == (1 / cpu_flops) * attention_flops(gbs, 1, s + n, nh, h1 / nh) * (bls // gbs) * (1 - gattn)
        prob += cpu_comm == (1 / c_bdw) * attention_bytes(gbs, 1, s + n, nh, nkvh, h1 / nh) * (bls // gbs) * (1 - gattn)
        
    ## Create variables for peak memory constraints
    gpu_home = pulp.LpVariable("gpu_home", lowBound=0)
    gpu_home_p = pulp.LpVariable("gpu_home^p", lowBound=0)
    gpu_home_d = pulp.LpVariable("gpu_home^d", lowBound=0)
    gpu_w = pulp.LpVariable("gpu_w^g", lowBound=0)

    inter = pulp.LpVariable("inter_gpu_working", lowBound=0)

    cpu_home = pulp.LpVariable("cpu_home^p", lowBound=0)
    cpu_w = pulp.LpVariable("cpu_w^p", lowBound=0)

    
    ## GPU peak memory constaints
    prob += gpu_home >= gpu_home_p
    prob += gpu_home >= gpu_home_d
    prob += gpu_home_d == w_other + wi * l * wg + 2 * 2 * h1 * bls + 4 * (s + n) * h_kv * bls * l * (cg + cc * gattn)
    # weights + gpu_experts + kv_cache_gpu + hidden_states_buffer (2gbs:hidden+residual) + kvcahe buffer (2gbs)
    # L4+padding
    # prob += gpu_home_p == w_other + wi * l * wg + 4 * s * h_kv * bls * cg + 2 * 2 * h1 * gbs * 2 * s + 4 * s * h_kv * gbs * 2 * cc
    # t4 (currently residual is not offloaded to cpu, so it is included in the peak memory to avoid OOM for T4)
    prob += gpu_home_p == w_other + wi * l * wg + 4 * s * h_kv * bls * cg + 2 * h1 * gbs * 2 * s + h1 * bls * 2 * s + 4 * s * h_kv * gbs * 2 * cc

    # for the fused moe kernel
    # if stage == "decode":
    #     prob += inter == gbs * topk * h1 + gbs * topk * 3 * h2
    # elif stage == "prefill":
    prob += inter == 2 * (gbs * topk * h1 * s + gbs * topk * 3 * h2 * s) * 1.2
    
    prob += gpu_w == 2 * wi * (1 - wg) + inter            
    prob += gpu_home + gpu_w <= gmem * 0.86

    ## CPU peak memory constraints
    # if stage == "decode":
    #     prob += cpu_home == wi * l * wc + 4 * (s + n) * h_kv * bls * l * cc
    # elif stage == "prefill":
    prob += cpu_home == wi * l * wc + 4 * (s + n) * h_kv * bls * l * cc + 2 * h1 * bls * s
    # pinned memory for weights
    prob += cpu_w == wi * (1 - wg) * 2
    prob += cpu_home + cpu_w <= cmem * 0.8

    # ------------ Finish add constraints ---------------

    ## Optimize model
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    if status == -1:
        return status, None, (0, -1), None, None

    gpu_peak = pulp.value(gpu_home) + pulp.value(gpu_w)

    cpu_peak = pulp.value(cpu_home) + pulp.value(cpu_w)

    t = pulp.value(T)
    if stage == "decode":
        t_tot = t * (n-1) * l
        throughput = bls * n / t_tot
    elif stage == "prefill":
        t_tot = t * l
        throughput = bls / t_tot

    ## print solution
    if verbose:
        print(f"status: {status}")
        print(f"weights size: {(w_other + wi * l) / GB:.4f} GB")
        print(f"ctog = {pulp.value(ctog):.4f} s  "
              f"gpu_T = {pulp.value(gpu_T):.4f} s  "
              f"cpu_T = {pulp.value(cpu_T):.4f} s")
        if stage == "prefill":
            print(f"gpu_comp_mlp = {pulp.value(gpu_comp_mlp):.4f} s  "
                f"gpu_comm_mlp = {pulp.value(gpu_comm_mlp):.4f} s  "
                f"gpu_comp_attn = {pulp.value(gpu_comp_attn):.4f} s "
                f"gpu_comm_attn = {pulp.value(gpu_comm_attn):.4f} s")
        print(f"T = {l * pulp.value(T):.3f} s")

        print(f"gpu peak mem ({stage}): {gpu_peak / GB:.3f} GB / {gmem * 0.9 / GB:.3f} GB")
        print(f"gpu_home: {pulp.value(gpu_home) / GB:.2f} GB")
        print(f"gpu_w: {pulp.value(gpu_w) / GB:.2f} GB")
        print(f"inter: {pulp.value(inter) / GB:.2f} GB")

        print(f"cpu peak mem ({stage}): {cpu_peak / GB:.3f} GB / {cmem * 0.8 / GB:.3f} GB")
        print(f"cpu_home_g: {pulp.value(cpu_home) / GB:.2f} GB")
        print(f"cpu_w_g: {pulp.value(cpu_w) / GB:.2f} GB")

        print(f"wg = {pulp.value(wg):.2f}  "
              f"wc = {pulp.value(wc):.2f}  ")
        print(f"cg = {pulp.value(cg):.2f}  "
              f"cc = {pulp.value(cc):.2f}  ")
        print(f"{stage} throughput = {throughput:.2f} token/s")
    
    policy = Policy(gbs, bls // gbs,
                    pulp.value(wg), pulp.value(wc),
                    pulp.value(cg), pulp.value(cc),
                    ne)
    return status, policy, (throughput, t_tot), gpu_peak, cpu_peak


def get_nb_ub(config, gbs, solve_lp):
    nb = 1
    while True:
        status, _, _, _, _ = solve_lp(config, gbs * nb, gbs, verbose=0)
        if status == -1: break
        nb *= 2

    left = max(nb // 2, 1)
    right = nb
    while left < right:
        mid = (left + right) // 2
        status, _, _, _, _ = solve_lp(config, gbs * mid, gbs, verbose=0)
        if status == 1:
            left = mid + 1
        elif status == -1:
            right = mid
    assert left == right
    nb_ub = left

    return nb_ub - 1


def best(policy1, throughput1, policy2, throughput2):
    if throughput2 >= throughput1:
        return policy2, throughput2
    else:
        return policy1, throughput1


def solve(model_config, hardware_config, args):
    config = CostModelConfig.init(model_config, hardware_config, args['prompt_len'], args['gen_len'])
    print(config)
    best_policy = None
    max_throughput = 0
    gbs = 1
    while True:
        if args["gbs"] is not None:
            gbs = args["gbs"]
        if args["num_gb"] is not None:
            status, policy, (throughput, _), _, _ = solve_lp(
                config, gbs * args["num_gb"], gbs, verbose=0, stage=args["stage"])
            if status == -1:
                break
            if status == 1:
                best_policy, max_throughput = best(best_policy, max_throughput, policy, throughput)
        else:
            nb_ub = get_nb_ub(config, gbs, solve_lp)
            if nb_ub == 0: break

            prev_throughput = 0
            for nb in range(1, nb_ub + 1):
                _, policy, (throughput, t_tot), _, _ = solve_lp(config, gbs * nb, gbs, verbose=0, stage=args["stage"])
                if throughput < prev_throughput:
                    break
                prev_throughput = throughput
                best_policy, max_throughput = best(best_policy, max_throughput, policy, throughput)
        if args["gbs"] is not None:
            break
        if gbs < 4:
            gbs += 1
        else:
            gbs += 32

    if best_policy is not None:
        _, _, _, _, _ = solve_lp(config, best_policy.ubs * best_policy.n_ub,
                           best_policy.ubs, verbose=True, stage=args["stage"])
    return best_policy, max_throughput


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-175b")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--gpu-mem", type=int, default=15)
    parser.add_argument("--cpu-mem", type=int, default=200)
    
    parser.add_argument("--gbs", "--gpu-batch-size", type=int)
    parser.add_argument("--num-gb", "--num-gpu-batches", type=int)
    parser.add_argument("--percent", nargs="+", type=int)
    parser.add_argument("--wg", type=int)
    parser.add_argument("--wc", type=int)
    parser.add_argument("--cg", type=int)
    parser.add_argument("--cc", type=int)
    parser.add_argument("--stage", type=str, default="decode")

    args = parser.parse_args()
    assert not (args.percent and (args.wg or args.wc or args.cg or args.cc))

    config = CostModelConfig()

    moe_config = ModelConfig(args.model)
    hardware_config = HardwareConfig()
    hardware_config.gmem = args.gpu_mem * GB
    hardware_config.cmem = args.cpu_mem * GB

    best_policy, max_throughput = solve(moe_config, hardware_config, vars(args))
    print(best_policy)
    print(f"max_throughput: {max_throughput:.8f} token/s")
 