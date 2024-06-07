from dataclasses import dataclass
from enum import Enum, auto
from typing import List

import numpy as np
import torch
from fastmoe.backend.memory import TokenToKVPool

class FinishReason(Enum):
    ABORT = auto()
    LENGTH = auto()
    EOS_TOKEN = auto()
    STOP_STR = auto()

class Req:
    def __init__(self, rid, input_text, input_ids):
        self.rid = rid
        self.input_text = input_text
        self.input_ids = input_ids
        self.input_len = len(input_ids)
        self.output_ids = []

        self.sampling_params = None
        self.return_logprob = False
        self.logprob_start_len = 0
        self.stream = False

        self.tokenizer = None
        self.finished = False
        self.finish_reason = None
        self.hit_stop_str = None

        self.logprob = None
        self.normalized_logprob = None

    def max_new_tokens(self):
        return self.sampling_params.max_new_tokens

    def check_finished(self):
        if self.finished:
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished = True
            self.finish_reason = FinishReason.LENGTH
            return

        if (
            self.output_ids[-1] == self.tokenizer.eos_token_id
            and self.sampling_params.ignore_eos == False
        ):
            self.finished = True
            self.finish_reason = FinishReason.EOS_TOKEN
            return
    
    def abort(self):
        self.finished = True
        self.finish_reason = FinishReason.ABORT

    def __repr__(self):
        return f"rid(n={self.rid}, " f"input_ids={self.input_ids}, "


@dataclass
class Batch:
    reqs: List[Req]
   
    token_to_kv_pool: TokenToKVPool = None
    cache_line_idx: int = None
    cpu_kv_pool_start_loc: int = None
    start_loc: torch.Tensor = None
    decode_out_cache_loc: torch.Tensor = None
    # intermediate results
    hidden_states: torch.Tensor = None
    hidden_states_cpu: torch.Tensor = None
    residual: torch.Tensor = None
    qkv: torch.Tensor = None

    # batched arguments to model runner
    input_ids: torch.Tensor = None
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    positions: torch.Tensor = None
    out_cache_loc: torch.Tensor = None

    start_loc_gpu: torch.Tensor = None
    new_num_tokens: int = None
    max_seq_len: int = None
    bs: int = None
    kv_pt: int = None

    return_logprob: bool = False

    # other arguments for control
    output_ids: torch.Tensor = None

    # batched sampling params
    temperatures: torch.Tensor = None
    top_ps: torch.Tensor = None
    top_ks: torch.Tensor = None
    frequency_penalties: torch.Tensor = None
    presence_penalties: torch.Tensor = None
    logit_bias: torch.Tensor = None

    @classmethod
    def init_new(cls, reqs):
        return_logprob = any(req.return_logprob for req in reqs)

        return cls(
            reqs=reqs,
            return_logprob=return_logprob,
        )

    def is_empty(self):
        return len(self.reqs) == 0
    
    def prepare_for_decode(self):
        # prepare for decode
        input_ids = [
            r.output_ids[-1] if r.output_ids else r.input_ids[-1] for r in self.reqs
        ]
        self.input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        self.seq_lens.add_(1)
        self.decode_out_cache_loc.add_(1)
        self.seq_lens_cpu = (self.seq_lens).to("cpu")

    
    def prepare_for_prefill(self, token_to_kv_pool: TokenToKVPool, vocab_size: int, int_token_logit_bias: torch.Tensor, max_output_len: int):
        device = "cuda"
        bs = len(self.reqs)
        reqs = self.reqs
        input_ids = [r.input_ids for r in reqs]
    
        flatten_input_ids = []
        seq_lens = []

        for i in range(bs):
            flatten_input_ids.extend(input_ids[i])
            seq_lens.append(len(input_ids[i]))

        # Alloc mem
        seq_lens = np.array(seq_lens)
        new_num_tokens = seq_lens.sum()
        print("new_num_tokens", new_num_tokens)

        bs = len(self.reqs)
        self.cpu_kv_pool_start_loc, gpu_kv_pool_start_loc, self.cache_line_idx = token_to_kv_pool.alloc_cpu()
        self.token_to_kv_pool = token_to_kv_pool
        
        # for kvcache offloading
        self.out_cache_loc = torch.zeros(new_num_tokens, device="cpu", dtype=torch.int64)
        self.decode_out_cache_loc = torch.zeros(bs, device="cpu", dtype=torch.int64)
        start_loc = torch.zeros((bs,), dtype=torch.int64, device="cpu")
        pt = 0
        out_cache_pt = 0
        for i in range(bs):
            seq_len = len(self.reqs[i].input_ids)
            self.out_cache_loc[out_cache_pt : out_cache_pt + seq_len] = torch.arange(gpu_kv_pool_start_loc + pt, gpu_kv_pool_start_loc + pt + seq_len, device="cpu", dtype=torch.int64)
            #  for cpu
            start_loc[i] = self.cpu_kv_pool_start_loc + pt
            self.decode_out_cache_loc[i] = self.cpu_kv_pool_start_loc + pt + seq_len - 1
            pt += seq_len + max_output_len
            out_cache_pt += seq_len
        self.kv_pt = self.decode_out_cache_loc[-1] + 1
        self.out_cache_loc = self.out_cache_loc.to('cuda')
        self.start_loc = start_loc

        # Handle logit bias
        logit_bias = torch.zeros((bs, vocab_size), dtype=torch.float32, device=device)
        for i in range(bs):
            if reqs[i].sampling_params.dtype == "int":
                logit_bias[i] = int_token_logit_bias

        # Set fields
        self.input_ids = torch.tensor(
            flatten_input_ids, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        self.start_loc_gpu = torch.zeros((bs,), dtype=torch.int64, device="cuda")
        self.start_loc_gpu[1:] = torch.cumsum(self.seq_lens[:-1], dim=0)
        self.positions = torch.cat([torch.arange(0, length, device='cuda') for length in seq_lens], dim=0)

        self.new_num_tokens = int(torch.sum(self.seq_lens))
        self.max_seq_len = int(torch.max(self.seq_lens))
        self.bs = bs

        self.temperatures = torch.tensor(
            [r.sampling_params.temperature for r in reqs],
            dtype=torch.float,
            device=device,
        ).view(-1, 1)
        self.top_ps = torch.tensor(
            [r.sampling_params.top_p for r in reqs], dtype=torch.float, device=device
        ).view(-1, 1)
        self.top_ks = torch.tensor(
            [r.sampling_params.top_k for r in reqs], dtype=torch.int, device=device
        ).view(-1, 1)
        self.frequency_penalties = torch.tensor(
            [r.sampling_params.frequency_penalty for r in reqs],
            dtype=torch.float,
            device=device,
        )
        self.presence_penalties = torch.tensor(
            [r.sampling_params.presence_penalty for r in reqs],
            dtype=torch.float,
            device=device,
        )
        self.logit_bias = logit_bias


    def offload_kv_cache(self, layer_id: int):
        store_size = self.kv_pt - self.cpu_kv_pool_start_loc
        self.token_to_kv_pool.store(self.cpu_kv_pool_start_loc, store_size, layer_id)
        
    
    def sample(self, logits: torch.Tensor):
        # Post process logits
        logits = logits.contiguous()
        logits.div_(self.temperatures)
        logits.add_(self.logit_bias)

        probs = torch.softmax(logits, dim=-1)
        probs_sort, probs_idx = _top_p_top_k(probs, self.top_ps, self.top_ks)
        sampled_index = torch.multinomial(probs_sort, num_samples=1)
        batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(
            -1
        )
        batch_next_token_probs = torch.gather(
            probs_sort, dim=1, index=sampled_index
        ).view(-1)

        return batch_next_token_ids, batch_next_token_probs


def _top_p_top_k(probs: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor):
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps] = 0.0
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1) >= top_ks
    ] = 0.0
    probs_sort.div_(probs_sort.max(dim=-1, keepdim=True)[0])
    return probs_sort, probs_idx
