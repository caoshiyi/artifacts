from dataclasses import dataclass
from enum import Enum, auto
from typing import List

import numpy as np
import torch
from fastmoe.backend.memory import ReqToTokenPool, TokenToKVPool


class ForwardMode(Enum):
    PREFILL = auto()
    DECODE = auto()
    PARTIAL = auto()
    PARTIAL_DECODE = auto()


class FinishReason(Enum):
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

        if len(self.sampling_params.stop_strs) > 0:
            tail_str = self.tokenizer.decode(
                self.output_ids[-(self.sampling_params.stop_str_max_len + 1) :]
            )

            for stop_str in self.sampling_params.stop_strs:
                if stop_str in tail_str:
                    self.finished = True
                    self.finish_reason = FinishReason.STOP_STR
                    self.hit_stop_str = stop_str
                    return

    def __repr__(self):
        return f"rid(n={self.rid}, " f"input_ids={self.input_ids}, "


@dataclass
class Batch:
    reqs: List[Req]
    req_to_token_pool: ReqToTokenPool
    req_to_cpu_token_pool: ReqToTokenPool
    token_to_kv_pool: TokenToKVPool

    # progress states, memory states
    cur_layer: int = 0
    cpu_indices: torch.Tensor = None
    all_cpu_indices: torch.Tensor = None
    gpu_indices: torch.Tensor = None
    req_cpu_pool_indices: torch.Tensor = None
    # intermediate results
    hidden_states: torch.Tensor = None
    residual: torch.Tensor = None

    # batched arguments to model runner
    input_ids: torch.Tensor = None
    req_pool_indices: torch.Tensor = None
    seq_lens: torch.Tensor = None
    position_ids_offsets: torch.Tensor = None
    out_cache_loc: torch.Tensor = None
    out_cache_cont_start: torch.Tensor = None
    out_cache_cont_end: torch.Tensor = None
    return_logprob: bool = False

    # other arguments for control
    output_ids: torch.Tensor = None
    new_num_tokens: int = None

    # batched sampling params
    temperatures: torch.Tensor = None
    top_ps: torch.Tensor = None
    top_ks: torch.Tensor = None
    frequency_penalties: torch.Tensor = None
    presence_penalties: torch.Tensor = None
    logit_bias: torch.Tensor = None

    @classmethod
    def init_new(cls, reqs, req_to_token_pool, token_to_kv_pool, req_to_cpu_token_pool=None):
        assert req_to_cpu_token_pool is not None
        return_logprob = any(req.return_logprob for req in reqs)

        return cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            req_to_cpu_token_pool=req_to_cpu_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            return_logprob=return_logprob,
        )

    def is_empty(self):
        return len(self.reqs) == 0
    
    def offload_kv_cache(self):
        # prefill first layer
        if self.cpu_indices is None:
            self.cpu_indices = self.token_to_kv_pool.offload_kv_cache(self.out_cache_loc)
            self.req_to_token_pool.free(self.req_pool_indices)
            self.req_pool_indices = None
            self.out_cache_loc = None
            self.req_cpu_pool_indices = self.req_to_cpu_token_pool.alloc(len(self.reqs))
            req_cpu_pool_indices_cpu = self.req_cpu_pool_indices.cpu().numpy()
            pt = 0
            for i in range(len(self.reqs)):
                seq_len = self.seq_lens[i].item()
                self.req_to_cpu_token_pool.req_to_token[req_cpu_pool_indices_cpu[i]][: seq_len] = self.cpu_indices[pt : pt + seq_len]
                pt += seq_len
            self.all_cpu_indices = self.cpu_indices
        # decode first layer
        elif self.cur_layer == 0:
            self.cpu_indices = self.token_to_kv_pool.offload_kv_cache(self.out_cache_loc)
            self.req_to_token_pool.free(self.req_pool_indices)
            self.req_pool_indices = None
            self.out_cache_loc = None
            req_cpu_pool_indices_cpu = self.req_cpu_pool_indices.cpu().numpy()
            all_cpu_indicies = []
            for i in range(len(self.reqs)):
                seq_len = self.seq_lens[i].item()
                self.req_to_cpu_token_pool.req_to_token[req_cpu_pool_indices_cpu[i]][seq_len] = self.cpu_indices[i]
                all_cpu_indicies.append(self.req_to_cpu_token_pool.req_to_token[req_cpu_pool_indices_cpu[i]][:seq_len+1])
            self.all_cpu_indices = torch.cat(all_cpu_indicies, dim=0)
        else:
            self.token_to_kv_pool.offload_kv_cache(self.out_cache_loc, self.cpu_indices, self.cur_layer)
            self.out_cache_loc = None

    def load_kv_cache(self, step_num_layers):
        assert self.all_cpu_indices is not None, "all_cpu_indices should not be None"
        bs = len(self.reqs)
        layer_ids = torch.arange(self.cur_layer, self.cur_layer + step_num_layers)
        if self.req_pool_indices is None:
            self.req_pool_indices = self.req_to_token_pool.alloc(bs)
            assert self.req_pool_indices is not None
        req_pool_indices_cpu = self.req_pool_indices.cpu().numpy()
        assert self.token_to_kv_pool.available_size() >= len(self.all_cpu_indices) * len(layer_ids) + bs * step_num_layers
        gpu_indices = self.token_to_kv_pool.load_kv_cache(self.all_cpu_indices, layer_ids)
        self.out_cache_loc = self.token_to_kv_pool.alloc(bs*step_num_layers).reshape(step_num_layers, -1)
        pt = 0
        for i in range(bs):
            seq_len = self.seq_lens[i].item()
            self.req_to_token_pool.req_to_token[req_pool_indices_cpu[i]][:step_num_layers, :seq_len] = gpu_indices[:, pt : pt + seq_len]
            self.req_to_token_pool.req_to_token[req_pool_indices_cpu[i]][:step_num_layers, seq_len] = self.out_cache_loc[:, i]
            pt += seq_len
    
    def prepare_for_partial_decode(self):
        # prepare for decode
        input_ids = [
            r.output_ids[-1] if r.output_ids else r.input_ids[-1] for r in self.reqs
        ]
        self.input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        self.seq_lens.add_(1)
        self.out_cache_cont_start = None
        self.out_cache_cont_end = None

    
    def prepare_for_partial_memory(self, step_num_layers):
        bs = len(self.reqs)
        req_pool_indices = self.req_to_token_pool.alloc(bs)
        req_pool_indices_cpu = req_pool_indices.cpu().numpy()
        self.out_cache_loc = self.token_to_kv_pool.alloc(self.new_num_tokens*step_num_layers).reshape(step_num_layers, -1)

        pt = 0
        for i in range(bs):
            seq_len = len(self.reqs[i].input_ids)
            self.req_to_token_pool.req_to_token[req_pool_indices_cpu[i]][:step_num_layers, :seq_len] = self.out_cache_loc[:, pt : pt + seq_len]
            pt += seq_len
        
        self.req_pool_indices = req_pool_indices
    
    def prepare_for_partial(self, vocab_size: int, int_token_logit_bias: torch.Tensor):
        device = "cuda"
        bs = len(self.reqs)
        reqs = self.reqs
        input_ids = [r.input_ids for r in reqs]
    
        flatten_input_ids = []
        seq_lens = []

        for i in range(bs):
            flatten_input_ids.extend(input_ids[i])
            seq_lens.append(len(input_ids[i]))

        position_ids_offsets = torch.zeros((bs,), dtype=torch.int32, device=device)

        # Alloc mem
        seq_lens = np.array(seq_lens)
        new_num_tokens = seq_lens.sum()
        print("new_num_tokens", new_num_tokens)

        # Handle logit bias
        logit_bias = torch.zeros((bs, vocab_size), dtype=torch.float32, device=device)
        for i in range(bs):
            if reqs[i].sampling_params.dtype == "int":
                logit_bias[i] = int_token_logit_bias

        # Set fields
        self.input_ids = torch.tensor(
            flatten_input_ids, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        self.position_ids_offsets = position_ids_offsets
        self.new_num_tokens = new_num_tokens

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

    def prepare_for_extend(self, vocab_size: int, int_token_logit_bias: torch.Tensor):
        device = "cuda"
        bs = len(self.reqs)
        reqs = self.reqs
        input_ids = [r.input_ids for r in reqs]
    
        flatten_input_ids = []
        seq_lens = []

        req_pool_indices = self.req_to_token_pool.alloc(bs)
        req_pool_indices_cpu = req_pool_indices.cpu().numpy()
        for i in range(bs):
            flatten_input_ids.extend(input_ids[i])
            seq_lens.append(len(input_ids[i]))

        position_ids_offsets = torch.zeros((bs,), dtype=torch.int32, device=device)

        # Alloc mem
        seq_lens = np.array(seq_lens)
        new_num_tokens = seq_lens.sum()
        out_cache_loc = self.token_to_kv_pool.alloc(new_num_tokens)

        pt = 0
        for i in range(bs):
            self.req_to_token_pool.req_to_token[req_pool_indices_cpu[i]][: seq_lens[i]] = out_cache_loc[pt : pt + seq_lens[i]]
            pt += seq_lens[i]

        # Handle logit bias
        logit_bias = torch.zeros((bs, vocab_size), dtype=torch.float32, device=device)
        for i in range(bs):
            if reqs[i].sampling_params.dtype == "int":
                logit_bias[i] = int_token_logit_bias

        # Set fields
        self.input_ids = torch.tensor(
            flatten_input_ids, dtype=torch.int32, device=device
        )
        self.req_pool_indices = req_pool_indices
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        self.position_ids_offsets = position_ids_offsets
        self.new_num_tokens = new_num_tokens
        self.out_cache_loc = out_cache_loc

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

    def check_decode_mem(self):
        bs = len(self.reqs)
        if self.token_to_kv_pool.available_size() >= bs:
            return True
        
        if self.token_to_kv_pool.available_size() >= bs:
            return True

        return False

    def retract_decode(self):
        sorted_indices = [i for i in range(len(self.reqs))]
        sorted_indices.sort(
            key=lambda i: (len(self.reqs[i].output_ids), -len(self.reqs[i].input_ids)),
            reverse=True,
        )

        retracted_reqs = []
        seq_lens_np = self.seq_lens.cpu().numpy()
        req_pool_indices_np = self.req_pool_indices.cpu().numpy()
        while self.token_to_kv_pool.available_size() < len(self.reqs):
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)

            req.output_ids = []


            token_indices = self.req_to_token_pool.req_to_token[
                req_pool_indices_np[idx]
            ][: seq_lens_np[idx]]
            self.token_to_kv_pool.free(token_indices)

        self.filter_batch(sorted_indices)

        return retracted_reqs

    def prepare_for_decode(self, input_ids=None):
        if input_ids is None:
            input_ids = [
                r.output_ids[-1] if r.output_ids else r.input_ids[-1] for r in self.reqs
            ]
        self.input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        self.seq_lens.add_(1)

        # Alloc mem
        bs = len(self.reqs)
        alloc_res = self.token_to_kv_pool.alloc_contiguous(bs)
        if alloc_res is None:
            self.out_cache_loc = self.token_to_kv_pool.alloc(bs)

            if self.out_cache_loc is None:
                print("Decode out of memory. This should nerver happen.")
                exit()

            self.out_cache_cont_start = None
            self.out_cache_cont_end = None
        else:
            self.out_cache_loc = alloc_res[0]
            self.out_cache_cont_start = alloc_res[1]
            self.out_cache_cont_end = alloc_res[2]

        self.req_to_token_pool.req_to_token[
            self.req_pool_indices, self.seq_lens - 1
        ] = self.out_cache_loc

    def filter_batch(self, unfinished_indices: List[int]):
        self.reqs = [self.reqs[i] for i in unfinished_indices]
        new_indices = torch.tensor(unfinished_indices, dtype=torch.int32, device="cuda")
        self.seq_lens = self.seq_lens[new_indices]
        self.input_ids = None
        self.req_pool_indices = self.req_pool_indices[new_indices]
        self.position_ids_offsets = self.position_ids_offsets[new_indices]
        self.out_cache_loc = self.out_cache_cont_start = self.out_cache_cont_end = None
        self.return_logprob = any(req.return_logprob for req in self.reqs)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "frequency_penalties",
            "presence_penalties",
            "logit_bias",
        ]:
            setattr(self, item, getattr(self, item)[new_indices])

    def merge(self, other):
        self.reqs.extend(other.reqs)

        self.req_pool_indices = torch.concat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = torch.concat([self.seq_lens, other.seq_lens])
        self.position_ids_offsets = torch.concat(
            [self.position_ids_offsets, other.position_ids_offsets]
        )
        self.out_cache_loc = self.out_cache_cont_start = self.out_cache_cont_end = None
        self.return_logprob = any(req.return_logprob for req in self.reqs)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "frequency_penalties",
            "presence_penalties",
            "logit_bias",
        ]:
            setattr(
                self, item, torch.concat([getattr(self, item), getattr(other, item)])
            )

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
