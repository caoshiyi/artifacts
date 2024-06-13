"""
An old script for fitting the cost model for OPT in FlexGen.

Warning:
The script has not been cleaned for release.
It has been placed here for study purposes only. There is no promise of reproduction.
"""

import argparse
import math
import os
import pickle as pkl
import torch
from torch import nn
import torch.nn.functional as F
from cases import suites
from flexgen.flex_moe import get_config

def get_filename(case):
    tokens = case.command.split()

    model_name = tokens[tokens.index('--model') + 1].split('/')[-1]  # Get the last part after '/'
    batch_size = int(tokens[tokens.index('--gpu-batch-size') + 1])
    num_gpu_batches = int(tokens[tokens.index('--num-gpu-batches') + 1]) if '--num-gpu-batches' in tokens else ""
    prompt_len = int(tokens[tokens.index('--prompt-len') + 1]) if '--prompt-len' in tokens else 76
    gen_len = int(tokens[tokens.index('--gen-len') + 1]) if '--gen-len' in tokens else 32  
    percent = tokens[tokens.index('--percent') + 1: tokens.index('--percent') + 7]  
    percent_str = '-'.join(percent)

    cpu_offload = '--cpu' in case.command
    disk_offload = case.use_page_maga if hasattr(case, 'use_page_maga') else False

    if "Mixtral" in model_name:
        model_name = "v0.1"
        
    filename = f"fo-{model_name}-gbs{batch_size}-ngbs{num_gpu_batches}-prompt{prompt_len}-gen{gen_len}-percent-{percent_str}-"

    if cpu_offload:
        filename += "cpu-cache.log"
    elif disk_offload:
        filename += "disk-cache.log"
    else:
        filename += "gpu-cache.log"
    
    return filename


# from experiments.run_exp import ExpConfig, cases, get_filename
from flexgen.utils import GB, T

class CostModel(nn.Module):
    def __init__(self, model="mistralai/Mixtral-8x22B-Instruct-v0.1"):
        super(CostModel, self).__init__()
        a = torch.abs(torch.rand([]))
        b = torch.abs(torch.rand([]))

        self.ctog_bdw        = nn.Parameter(a)
        self.gtoc_bdw_cache  = nn.Parameter(a)
        self.gtoc_bdw_hidden = nn.Parameter(a)

        self.dtoc_bdw          = nn.Parameter(a)
        self.ctod_bdw_cache_p  = nn.Parameter(a)
        self.ctod_bdw_hidden_p = nn.Parameter(a)
        self.ctod_bdw_g        = nn.Parameter(a)

        self.mm_flops_p  = nn.Parameter(b)
        self.mm_flops_g  = nn.Parameter(b)
        self.bmm_flops_p = nn.Parameter(b)
        self.bmm_flops_g = nn.Parameter(b)
        self.cpu_flops   = nn.Parameter(b)

        self.c0 = nn.Parameter(torch.tensor(0.0))
        self.c1 = nn.Parameter(torch.tensor(0.0))
        self.c2 = nn.Parameter(torch.tensor(0.0))
        self.c3 = nn.Parameter(torch.tensor(0.0))


    def forward(self, xs):
        (wi, l, h1, h2, s, n,
         gbs, bls, wg, wc, wn, cg, cc, cn, hg, hc, hn, nh, nkvh, ne, h_kv) = xs.split(1, dim=1)
                
        # this is a bit different than 1 / ctog_bdw in cost_model_moe.py?
        ctogp = (self.ctog_bdw / GB) * (wi * (wc + wn)
                 + 2 * s * h1 * bls * (hc + hn))
        
        gtocp = (self.gtoc_bdw_cache / GB) * (4 * (s + 1) * h_kv * bls * (cc + cn)) \
                 + (self.gtoc_bdw_hidden / GB) * 2 * s * h1 * bls * (hc + hn)
                 
        dtocp = (self.dtoc_bdw / GB) * (wi * wn + 2 * s * h1 * bls * hn)
        
        ctodp = (self.ctod_bdw_cache_p / GB) * 4 * bls * (s + 1) * h_kv * cn \
                  + (self.ctod_bdw_hidden_p / GB) * 2 * s * h1 * bls * hn
                  
        compp = (self.mm_flops_p / T) * bls * (4 * s * h1 ** 2 + 4 * s * h1 * h_kv  + 6 * s * h1 * h2) \
                  + (self.bmm_flops_p / T) * 4 * bls * s ** 2 * h1
                  
        tpre = torch.maximum(ctogp + dtocp, torch.maximum(gtocp + ctodp, compp))

        ctogg = (self.ctog_bdw / GB) * (wi * (wc + wn)
                  + 2 * h1 * bls * (hc + hn))
        
        gtocg = (self.gtoc_bdw_hidden / GB) * 2 * h1 * bls * (hc + hn)
        
        dtocg = (self.dtoc_bdw / GB) * (4 * bls * (s + n / 2) * h_kv * cn
                                        + 2 * h1 * bls * hn) \
                  + (self.dtoc_bdw / GB / 0.95) * wi * wn 
                  
        ctodg = (self.ctod_bdw_g / GB) * (4 * bls * h_kv * cn
                                               + 2 * h1 * bls * hn)


        # non-linear cpu_flops
        cpu_flops_real = self.cpu_flops / torch.clamp(
            1 + self.c1 * torch.log2(64 / gbs).clamp(min=0) * torch.log2(4096 / h1).clamp(min=0)
            - self.c2 * torch.log2(64 / gbs).clamp(min=0)
            - self.c3 * torch.log2(4096 / h1).clamp(min=0),
            min=0.5)
        compg = (self.mm_flops_g / T) * bls * (4 * h1 ** 2 + 4 * h1 * h_kv + 6 * h1 * h2) \
             + (self.bmm_flops_g / T) * 4 * bls * (s + n / 2) * h1 * cg \
             + (cpu_flops_real / T) * 4 * bls * (s + n / 2) * h1 * (cc + cn)

        tgen = ctogg + torch.maximum(gtocg,
                                     torch.maximum(dtocg,
                                     torch.maximum(ctodg, compg)))
        return torch.cat([tpre * l, tgen * (n - 1) * l], dim=1)

import re

def parse_filename(filename):
    pattern = r"fo-v0.1-gbs(\d+)-ngbs(\d+)-prompt(\d+)-gen(\d+)-percent-.*\.log"
    match = re.match(pattern, filename)
    if match:
        gbs = int(match.group(1))
        bls = int(match.group(2))
        prompt_length = int(match.group(3))
        gen_length = int(match.group(4))
        return gbs, bls, prompt_length, gen_length
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-dir", type=str, default="../experiments/results")
    args = parser.parse_args()
    torch.manual_seed(0)

    model = CostModel(model="mistralai/Mixtral-8x22B-Instruct-v0.1")
    model_name = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    model.double()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-3)

    dataset = []
    log_directory = "/home/cusgadmin/FlexGen"  # directory containing the log files

    filenames = [f for f in os.listdir(log_directory) if f.endswith('.log')]

    dataset = []
    for filename in filenames:
        match = re.match(r"fo-v0\.1-gbs(\d+)-ngbs(\d+)-prompt(\d+)-gen(\d+)-percent-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(.+?)\.log", filename)
        if match:
            gbs, bls, prompt_length, gen_length, wg, wc, cg, cc, hg, hc = map(int, match.groups()[:10])
            case_name = match.group(11)

            # Constructing stats directly from filename
            stats = {'gbs': gbs, 'bls': bls, 'prompt_len': prompt_length, 'gen_len': gen_length,
                     'percent': (wg, wc, cg, cc, hg, hc)}

            try:
                with open(os.path.join(log_directory, filename), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(":", 1)
                            stats[key.strip()] = float(value.split()[0])
            except FileNotFoundError:
                # Handling file not found, if necessary
                continue 
            
            prefill_latency = stats.get('prefill latency') 
            decode_latency = stats.get('decode latency') 
            print(f"filename: {filename}, prefill_latency: {prefill_latency}, decode_latency: {decode_latency}")
            
            s = prompt_length
            n = gen_length
            moe_config = get_config(model_name)
            # moe_config = get_moe_config(case.model_name)
            l = moe_config.num_hidden_layers
            h1 = moe_config.hidden_size
            h2 = moe_config.intermediate_size
            nh = moe_config.num_attention_heads
            nkvh = moe_config.num_key_value_heads
            ne = moe_config.num_local_experts
            num_kv_group = nh // nkvh
            h_kv = h1 // num_kv_group
            # NOTE: this is updated to fit moe in cost model 
            wi = 4 * h1 ** 2 + 4 * h1 ** 2 // num_kv_group + 6 * h1 * h2 * ne
            
            wn = 100 - wg - wc
            cn = 100 - cg - cc
            hn = 100 - hg - hc
            wg, wc, wn, cg, cc, cn, hg, hc, hn = (
                wg / 100, wc / 100, wn / 100, cg / 100, cc / 100,
                cn / 100, hg / 100, hc / 100, hn  / 100)
            x = torch.tensor([[wi, l, h1, h2, s, n,
                            gbs, bls, wg, wc, wn, cg, cc, cn, hg, hc, hn, nh, nkvh, ne, h_kv]])
        
            y = torch.tensor([[prefill_latency, decode_latency]])
            dataset.append((x, y))

    xs = torch.cat([row[0] for row in dataset])
    ys = torch.cat([row[1] for row in dataset])

    indices = torch.randperm(xs.shape[0])
    xs = xs[indices]
    ys = ys[indices]
    split = int(0.9 * len(xs))

    xs_train, xs_test = xs[:split], xs[split:]
    ys_train, ys_test = ys[:split], ys[split:]

    def compute_loss(xs, ys):
        ys_pred = model(xs)
        return loss_fn(ys_pred / ys, torch.ones_like(ys))

    num_epoches = 30000

    def set_update_cpu_delta(flag):
        model.c0.requires_grad = flag
        model.c1.requires_grad = flag
        model.c2.requires_grad = flag
        model.c3.requires_grad = flag

    def freeze_all_params():
        for param in model.parameters():
            param.requires_grad = False

    set_update_cpu_delta(False)

    for i in range(num_epoches):
        reg_loss = compute_loss(xs_train, ys_train)
        penalty_loss = 0
        for name, p in model.named_parameters():
            penalty_loss += F.relu(-p)
        penalty_loss += F.relu(model.gtoc_bdw_hidden - model.gtoc_bdw_cache)
        penalty_loss += F.relu(model.mm_flops_p - model.bmm_flops_p)
        penalty_loss += F.relu(model.mm_flops_g - model.bmm_flops_g)
        penalty_loss += F.relu(model.bmm_flops_p - model.bmm_flops_g)
        penalty_loss += F.relu(model.mm_flops_p - model.mm_flops_g)

        loss = reg_loss + penalty_loss * 100
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i == int(num_epoches * 0.8):
            freeze_all_params()
            set_update_cpu_delta(True)
            optimizer.param_groups[0]['lr'] = 1e-3

        if i % 200 == 0:
            eval_loss = compute_loss(xs_train, ys_train)
            print(f"epoch: {i}, train_loss: {loss.item():.6f}, "
                  f"eval_loss: {eval_loss.item():.6f}")

        #for name, p in model.named_parameters():
            #print(name, p.grad)

    for name, param in model.named_parameters():
        if "bdw" in name:
            print(f"{name}:\t {1 / param.item():.4f} GB/s")
        elif "flops" in name:
            print(f"{name}:\t {1 / param.item():.4f} T")
        elif "c" in name:
            print(f"{name}:\t {param.item():.4f}")

    print(f"len(dataset): {len(dataset)}")
