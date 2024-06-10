import torch
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

from fastmoe.utils.model_config import ModelConfig
from fastmoe.backend.memory import TokenToKVPool
from fastmoe.backend.optimizer import solve, Policy
from fastmoe.backend.task import Batch, Req
from fastmoe.backend.task_meta import ForwardMode, DecodePart
from fastmoe.backend.utils import HardwareConfig
from fastmoe.backend.model_runner import ModelRunner


class ExecutionEngine:
    def __init__(self, model_runner: ModelRunner, model_config: ModelConfig, hardware_config: HardwareConfig, avg_prompt_len: int, gen_len: int):
        self.model_runner = model_runner
        self.model_config = model_config
        self.hardware_config = hardware_config
        self.context = ExecutionContext.build_context(model_config, hardware_config, avg_prompt_len, gen_len)
        self.micro_batches: List[Batch] = None

        # sync primititves
        # record in prefetch experts from pin, wait in prefetch experts to pin & post attn
        self.prefetch_events: List[torch.cuda.Event] = None
        # sets in prefill attn, wait in offload_kv_cache
        self.attn_events: List[torch.cuda.Event] = None
        # sets after preattn, wait in offload_qkv
        self.compute_events: List[torch.cuda.Event] = None
        # wait for preill_load hidden in layer
        self.prefill_load_hidden_events: List[torch.cuda.Event] = None
        # wait for pre_attn
        self.offload_qkv_events: List[torch.cuda.Event] = None
        # wait for cpu attn
        self.load_hidden_events: List[torch.cuda.Event] = None
        # wait for cpu_attention in post_attention
        self.attn_futures = None
        # wait for copy to pin in prefetch from pin
        self.copy_futures = None

        # task meta data
        self.weights_prefetch_page_gpu = 0
        self.weights_prefetch_page_cpu = 0
        self.weights_prefetch_page_pin = 0
        self.weights_prefetch_num_pages_cpu = self.model_config.num_local_experts // self.context.policy.eg
        self.weights_prefetch_num_pages_gpu = 2

        # experts cache mapping: experts_id -> idx in self.context.experts_cache
        self.experts_mapping = [torch.empty(self.model_config.num_local_experts, dtype=torch.int64, device="cuda") for _ in range(self.model_config.num_hidden_layers)]
    
    def init_weights_prefetch_meta(self):

        # weights prefetching
        self.num_comp_experts = self.context.policy.eg
        self.num_experts_gpu  = int(self.model_config.num_local_experts * self.context.policy.wg)

        # prefill meta
        self.num_weights_slots_prefill = 1
        self.page_size = self.num_comp_experts - self.num_experts_gpu
        self.prefill_slot_size = self.page_size
        
        #  fine-grained page size for decode
        self.num_weights_slots_decode = len(self.micro_batches)
        self.fg_page_size = self.page_size * 3 * self.model_config.hidden_size
        self.decode_slot_size = self.fg_page_size // self.num_weights_slots_decode
    
    def _get_prefetch_cpu_slice(self, slot_id: int, stage: str):
        page_id = self.weights_prefetch_page_cpu
        if stage == 'prefill':
            cpu_start_pos = (self.num_experts_gpu 
                    + page_id * self.page_size 
                    + slot_id * self.prefill_slot_size)
            self.weights_prefetch_page_cpu = (self.weights_prefetch_page_cpu + 1) % self.weights_prefetch_num_pages_cpu
            return slice(cpu_start_pos, cpu_start_pos + self.prefill_slot_size)
        elif stage == 'decode':
            cpu_start_pos = (self.num_experts_gpu * 3 * self.model_config.hidden_size 
                    + page_id * self.fg_page_size 
                    + slot_id * self.decode_slot_size)
            if slot_id == self.num_weights_slots_decode - 1:
                self.weights_prefetch_page_cpu = (self.weights_prefetch_page_cpu + 1) % self.weights_prefetch_num_pages_cpu
                return slice(cpu_start_pos, None)
            else:
                return slice(cpu_start_pos, cpu_start_pos + self.decode_slot_size)
    
    def _get_prefetch_gpu_slice(self, slot_id: int, stage: str):
        page_id = self.weights_prefetch_page_gpu
        if stage == 'prefill':
            gpu_start_pos = (self.context.get_ecache_size() 
                        - self.weights_prefetch_num_pages_gpu * self.page_size 
                        + page_id * self.page_size 
                        + slot_id * self.prefill_slot_size)
            # update gpu page pointers
            self.weights_prefetch_page_gpu = (self.weights_prefetch_page_gpu + 1) % self.weights_prefetch_num_pages_gpu
            return slice(gpu_start_pos, gpu_start_pos + self.prefill_slot_size)
        elif stage == 'decode':
            gpu_start_pos = (self.context.get_ecache_size() * 3 * self.model_config.hidden_size 
                        - self.weights_prefetch_num_pages_gpu * self.fg_page_size 
                        + page_id * self.fg_page_size 
                        + slot_id * self.decode_slot_size)
            # update gpu page pointers
            if slot_id == self.num_weights_slots_decode - 1:
                self.weights_prefetch_page_gpu = (self.weights_prefetch_page_gpu + 1) % self.weights_prefetch_num_pages_gpu
                remain = self.fg_page_size - slot_id * self.decode_slot_size
                return slice(gpu_start_pos, gpu_start_pos + remain)
            else:
                return slice(gpu_start_pos, gpu_start_pos + self.decode_slot_size)


    
    def create_micro_batches(self, req_queue: List[Req]):     
        abort_requests: List[Req] = []
        micro_batches = []

        # for warmup
        if len(req_queue) < self.context.policy.ubs:
            micro_batches.append(Batch.init_new(req_queue))
        else:
            partitions = [[] for _ in range(self.context.policy.n_ub)]

            # Initialize partitions
            partitions_sums = [0] * self.context.policy.n_ub
            
            # Sort numbers in descending order for better distribution in greedy approach
            prompt_len_sorted = sorted(req_queue, key=lambda x: x.input_len, reverse=True)
            
            # Greedily assign each number to the partition with the minimum sum
            for req in prompt_len_sorted:
                if len(partitions) == 0:
                    abort_requests.append(req)
                    continue

                # Find the partition with the smallest current sum
                idx = partitions_sums.index(min(partitions_sums))
                if ((partitions_sums[idx] + req.input_len) 
                    + (1 + len(partitions[idx])) * self.context.gen_len 
                    > self.context.token_to_kv_pool.cache_line):
                    # if no partition can hold the req, abort the req
                    abort_requests.append(req)
                    continue
                else:
                    partitions[idx].append(req)
                    partitions_sums[idx] += req.input_len
                    if len(partitions[idx]) == self.context.policy.ubs:
                        micro_batches.append(Batch.init_new(partitions[idx]))
                        partitions.pop(idx)
                        partitions_sums.pop(idx)
            
            # abort requests
            for req in abort_requests:
                req.abort()

            for mb in partitions:
                micro_batches.append(Batch.init_new(mb))
        
        # self.num_weights_slots_decode = len(micro_batches)
        self.micro_batches = micro_batches

        self.prefetch_events = [torch.cuda.Event() for _ in range(len(micro_batches))]
        self.attn_events = [torch.cuda.Event() for _ in range(len(micro_batches))]
        self.compute_events = [torch.cuda.Event() for _ in range(len(micro_batches))]
        self.prefill_load_hidden_events = [torch.cuda.Event() for _ in range(len(micro_batches))]
        self.offload_qkv_events = [torch.cuda.Event() for _ in range(len(micro_batches))]
        self.load_hidden_events = [torch.cuda.Event() for _ in range(len(micro_batches))]

        self.attn_futures = [None for _ in range(len(micro_batches))]
        self.copy_futures = [None for _ in range(len(micro_batches))]
            
        return len(micro_batches), abort_requests
    
    def prepare_for_prefill(self, int_token_logit_bias: torch.Tensor):
        for micro_batch in self.micro_batches:
            micro_batch.prepare_for_prefill(token_to_kv_pool=self.context.token_to_kv_pool,
                                            vocab_size=self.model_config.vocab_size,
                                            int_token_logit_bias=int_token_logit_bias,
                                            max_output_len=self.context.gen_len,
                                            )
    
    def prepare_for_decode(self):
        for micro_batch in self.micro_batches:
            micro_batch.prepare_for_decode()
    
    def prefetch_experts(self, layer_id: int, page_id: int = 0, slot_id: int = 0, stage = 'prefill'):
        # do not need chunked weights
        if stage == 'prefill':
            assert(slot_id == 0)
            assert(page_id == self.weights_prefetch_page_cpu)
            prefetch_gpu_slice = self._get_prefetch_gpu_slice(slot_id, stage)
            prefetch_cpu_slice = self._get_prefetch_cpu_slice(slot_id, stage)
            with torch.cuda.stream(self.context.prefetch_stream):
                self.context.experts_cache[prefetch_gpu_slice].copy_(self.model_runner.model.get_experts_mem()[layer_id, prefetch_cpu_slice], non_blocking=True)
                self.prefetch_events[slot_id].record(self.context.prefetch_stream)
        elif stage == 'decode':
            # wait on copy to pin
            self.copy_futures[slot_id].result()
            assert(slot_id < self.num_weights_slots_decode) 
            prefetch_gpu_slice = self._get_prefetch_gpu_slice(slot_id, stage)
            if slot_id != self.num_weights_slots_decode - 1:
                from_pin_slice = slice(
                    slot_id * self.decode_slot_size, 
                    (slot_id + 1) * self.decode_slot_size
                )
            else:
                from_pin_slice = slice(
                    slot_id * self.decode_slot_size, 
                    None
            )
            with torch.cuda.stream(self.context.load_stream):
                self.context.experts_cache.view(-1, self.model_config.intermediate_size)[prefetch_gpu_slice, :].copy_(self.context.experts_pin.view(-1, self.model_config.intermediate_size)[from_pin_slice, :], non_blocking=True)
                self.prefetch_events[slot_id].record(self.context.prefetch_stream)
    
    def prefetch_experts_to_pin(self, layer_id: int, slot_id: int):
        self.copy_futures[slot_id] = self.context.copy_executor.submit(self.prefetch_experts_to_pin_func, layer_id, slot_id)

    def prefetch_experts_to_pin_func(self, layer_id: int, slot_id: int):
        self.prefetch_events[slot_id].synchronize()

        experts_pin = self.context.experts_pin.view(-1, 
                                                    self.model_config.intermediate_size)
        experts_cpu = (self.model_runner.model.get_experts_mem()
                       .view(self.model_config.num_hidden_layers, 
                            -1, 
                            self.model_config.intermediate_size))
        prefetch_cpu_slice = self._get_prefetch_cpu_slice(slot_id, 'decode')
        if slot_id != self.num_weights_slots_decode - 1:
            to_pin_slice = slice(
                slot_id * self.decode_slot_size, 
                (slot_id + 1) * self.decode_slot_size
            )
        else:
            to_pin_slice = slice(
                slot_id * self.decode_slot_size, 
                None
            )

        experts_pin[to_pin_slice, :].copy_(experts_cpu[layer_id, prefetch_cpu_slice, :])
        print(f"Prefetch experts to pin: layer {layer_id}, slot {slot_id}")
    
    def offload_kv_cache(self, layer_id: int, batch_id: int):
        with torch.cuda.stream(self.context.offload_stream):
            self.attn_events[batch_id].wait(self.context.offload_stream)
            self.micro_batches[batch_id].offload_kv_cache(layer_id=layer_id)
            cache_line_idx = self.micro_batches[batch_id].cache_line_idx
            self.context.offload_kv_events[cache_line_idx].record(self.context.offload_stream)
    
    def load_hidden_prefill(self, batch_id: int):
        if self.micro_batches[batch_id].hidden_states_cpu is not None:
            with torch.cuda.stream(self.context.load_stream):
                self.compute_events[batch_id].wait(self.context.load_stream)
                self.micro_batches[batch_id].hidden_states = self.micro_batches[batch_id].hidden_states_cpu.to('cuda', non_blocking=True)
                self.prefill_load_hidden_events[batch_id].record(self.context.load_stream)
    
    def offload_hidden_prefill(self, batch_id: int):
        # wait on compute event:
        if self.micro_batches[batch_id].hidden_states is not None:
            with torch.cuda.stream(self.context.offload_stream):
                self.compute_events[batch_id].wait(self.context.offload_stream)
                if self.micro_batches[batch_id].hidden_states_cpu is None:
                    self.micro_batches[batch_id].hidden_states_cpu = torch.empty_like(self.micro_batches[batch_id].hidden_states, device="cpu").pin_memory()
                self.micro_batches[batch_id].hidden_states_cpu.copy_(self.micro_batches[batch_id].hidden_states, non_blocking=True)
                self.micro_batches[batch_id].hidden_states = None

    # read: weights, write: kvcache, 
    def layer(self, layer_id: int, page_id: int, batch_id: int):
        print("Prefill layer: ", layer_id, "batch: ", batch_id)
        # wait on prefetch weights, load hidden and offload kv cache of the same cacheline
        batch = self.micro_batches[batch_id]
        if batch_id == 0:
            self.prefetch_events[self.num_weights_slots_prefill - 1].wait(self.context.cur_stream)
        self.context.offload_kv_events[batch.cache_line_idx].wait(self.context.cur_stream)
        self.prefill_load_hidden_events[batch_id].wait(self.context.cur_stream)
        
        #  computation
        if self.context.policy.eg != self.model_config.num_local_experts:
            # todo: add expert-level computation
            pass
        else:
            if layer_id < self.model_config.num_hidden_layers - 1:
                batch.hidden_states, batch.residual = self.model_runner.prefill(
                    batch, DecodePart.ALL, 
                    layer_id, self.experts_mapping[layer_id],
                    batch.return_logprob, self.attn_events[batch_id]
                )
            else:
                # last layer forward
                logits, (logprobs, normalized_logprobs) = self.model_runner.prefill(
                    batch, DecodePart.ALL, layer_id,
                    self.experts_mapping[layer_id],
                    batch.return_logprob, self.attn_events[batch_id]
                )
                if logprobs is not None:
                    logprobs = logprobs.cpu().tolist()
                    normalized_logprobs = normalized_logprobs.cpu().tolist()

                next_token_ids, next_token_probs = batch.sample(logits)
                next_token_ids = next_token_ids.cpu().tolist()
                print("Next token ids: ", next_token_ids)
                batch.hidden_states = None
                batch.hidden_states_cpu = None
                batch.residual = None

                reqs = batch.reqs
                pt = 0
                for i, req in enumerate(reqs):
                    req.output_ids = [next_token_ids[i]]
                    req.check_finished()

                    if logprobs is not None:
                        req.logprob = logprobs[pt : pt + req.input_len - 1]
                        req.normalized_logprob = normalized_logprobs[i]
                        pt += req.input_len
        # record compute event
        self.compute_events[batch_id].record()

    def pre_attention(self, layer_id: int, batch_id: int):
        batch = self.micro_batches[batch_id]
        batch.qkv, batch.residual = self.model_runner.pre_attn(
                batch, DecodePart.PREATTN, layer_id
            )
        self.compute_events[batch_id].record()
    
    def offload_qkv(self, layer_id: int, batch_id: int):
        with torch.cuda.stream(self.context.offload_stream):
            self.compute_events[batch_id].wait(self.context.offload_stream)
            bs = self.micro_batches[batch_id].bs
            self.context.qkv_pin[batch_id][ :bs, :].copy_(self.micro_batches[batch_id].qkv[:, :], non_blocking=True)
            self.offload_qkv_events[batch_id].record(self.context.offload_stream)

    def cpu_attention(self, layer_id: int, batch_id: int):
        self.attn_futures[batch_id] = self.context.cpu_executor.submit(self.cpu_attetion_func, layer_id, batch_id)
    
    def cpu_attetion_func(self, layer_id: int, batch_id: int):
        batch = self.micro_batches[batch_id]
        bs = batch.bs
        self.offload_qkv_events[batch_id].synchronize()
        self.model_runner.cpu_attention(
                batch, DecodePart.CPU_ATTN, layer_id, 
                self.context.qkv_pin[batch_id][ :bs, :], 
                self.context.hidden_pin[batch_id][ :bs, :, :, :],
            )
    
    def load_hidden(self, layer_id: int, batch_id: int):
        self.attn_futures[batch_id].result()
        with torch.cuda.stream(self.context.load_stream):
            bs = self.micro_batches[batch_id].bs
            self.micro_batches[batch_id].hidden_states = self.context.hidden_pin[batch_id][ :bs, :, :, :].to('cuda', non_blocking=True)
            self.load_hidden_events[batch_id].record(self.context.load_stream)

    def post_attention(self, layer_id: int, page_id: int, batch_id: int):
        batch = self.micro_batches[batch_id]
        if batch_id == 0:
            self.prefetch_events[self.num_weights_slots_decode - 1].wait(self.context.cur_stream)
        self.load_hidden_events[batch_id].wait(self.context.cur_stream)
        if layer_id < self.model_config.num_hidden_layers - 1:
            batch.hidden_states, batch.residual = self.model_runner.post_attn(
                batch, DecodePart.POSTATTN, layer_id, self.experts_mapping[layer_id]
            )
        else:
            # last layer forward
            logits, _ = self.model_runner.post_attn(batch, 
                                                  DecodePart.POSTATTN, 
                                                  layer_id, 
                                                  self.experts_mapping[layer_id])
            next_token_ids, next_token_probs = batch.sample(logits)
            next_token_ids = next_token_ids.cpu().tolist()
            print("Next token ids: ", next_token_ids)
            batch.hidden_states = None
            batch.residual = None

            # Check finish condition
            reqs = batch.reqs
            for i in range(len(reqs)):
                reqs[i].output_ids.append(next_token_ids[i])
                reqs[i].check_finished()
    
    # deal with corner cases for weights prefetching to pin
    def get_prefetch_e2p_idx(self, i, j, k, num_bs):
        if self.weights_prefetch_num_pages_cpu == 1:
            assert j == 0
            if i + 1 < self.model_config.num_hidden_layers:
                if k + 2 < num_bs:
                    return (i + 1) % self.model_config.num_hidden_layers, k + 2
                else:
                    return (i + 2) % self.model_config.num_hidden_layers, (k + 2) % num_bs
            else:
                if k + 2 < num_bs:
                    return (i + 1) % self.model_config.num_hidden_layers, k + 2
                else:
                    return None, None
        elif self.weights_prefetch_num_pages_cpu > 1:
            if j + 1 < self.weights_prefetch_num_pages_cpu:
                if k + 2 < num_bs:
                    return i, k + 2
                elif k + 2 >= num_bs and j + 2 < self.weights_prefetch_num_pages_cpu:
                    return i, (k + 2) % num_bs
                elif k + 2 >= num_bs and j + 2 == self.weights_prefetch_num_pages_cpu:
                    return (i + 1) % self.model_config.num_hidden_layers, (k + 2) % num_bs
            else:
                if k + 2 < num_bs:
                    return (i + 1) % self.model_config.num_hidden_layers, k + 2
                elif k + 2 >= num_bs:
                    return None, None

    def init_gpu_experts(self):
        self.context.init_gpu_experts(self.model_runner.model.get_experts_mem())
        # link the experts cache to the model
        self.model_runner.model.link_gpu_experts_cache(self.context.experts_cache)
        num_gpu_experts = int(self.model_config.num_local_experts * self.context.policy.wg)
        if num_gpu_experts > 0:
            for i in range(self.model_config.num_hidden_layers):
                self.experts_mapping[i][:num_gpu_experts] = torch.arange(i * num_gpu_experts, (i + 1) * num_gpu_experts, dtype=torch.int64, device="cuda")
        
        assert self.model_config.num_hidden_layers % 2 == 0
        
        num_comp_experts = self.context.policy.eg
        page_size = num_comp_experts - num_gpu_experts
        for i in range(self.model_config.num_hidden_layers):
            for page_id in range(self.weights_prefetch_num_pages_cpu):
                start_pos = num_gpu_experts + page_id * page_size
                gpu_page_id = (i * self.weights_prefetch_num_pages_cpu + page_id) % self.weights_prefetch_num_pages_gpu
                start_pos_cache = self.context.get_ecache_size() - self.weights_prefetch_num_pages_gpu * page_size + gpu_page_id * page_size
                self.experts_mapping[i][start_pos : start_pos + page_size] = torch.arange(start_pos_cache, start_pos_cache + page_size, dtype=torch.int64, device="cuda")

    def delete_gpu_context(self):
        self.context.delete_gpu_context()
    
    def reset(self):
        self.micro_batches = None
        self.prefetch_events = None
        self.attn_events = None
        self.compute_events = None
        self.prefill_load_hidden_events = None
        self.attn_futures = None
        self.copy_futures = None
        self.weights_prefetch_page_gpu = 0
        self.weights_prefetch_page_cpu = 0
        self.weights_prefetch_page_pin = 0
        self.weights_prefetch_num_pages_cpu = self.model_config.num_local_experts // self.context.policy.eg
        self.weights_prefetch_num_pages_gpu = 2
        self.num_weights_slots_prefill = 1
        self.num_weights_slots_decode = 0

        self.context.token_to_kv_pool.clear()


@dataclass
class ExecutionContext:
    model_config: ModelConfig
    policy: Policy
    token_to_kv_pool: TokenToKVPool

    # experts cache on gpu
    experts_cache: torch.tensor

    # cpu pinned relay
    qkv_pin: List[torch.tensor]
    hidden_pin: List[torch.tensor]
    experts_pin: torch.tensor

    # multi-thread executors and futures
    cpu_executor: ThreadPoolExecutor
    copy_executor: ThreadPoolExecutor
    prefetch_stream: torch.cuda.stream
    offload_stream: torch.cuda.stream
    load_stream: torch.cuda.stream
    cur_stream: torch.cuda.stream

    offload_kv_events: List[torch.cuda.Event]

    # meta data
    avg_prompt_tokens: int
    gen_len: int
    
    @classmethod
    def build_context(cls, model_config: ModelConfig, hardware_config: HardwareConfig, avg_prompt_len: int, gen_len: int):
        avg_prompt_tokens = avg_prompt_len
        max_new_tokens = gen_len

        # searching for optimal micro batch size and batch size
        opt_args = {}
        opt_args['prompt_len'] = avg_prompt_tokens
        opt_args['gen_len'] = max_new_tokens
        opt_args['stage'] = 'decode'
        opt_args['gbs'] = None
        opt_args['num_gb'] = None
        
        policy, _ = solve(model_config, hardware_config, opt_args)
        print(f"Policy: {policy}")
        # # hack
        # policy.ubs = 55
        # policy.n_ub = 5
        
        # allocate mem for the context
        ubs = policy.ubs
        bs = policy.n_ub * ubs
        # kv cache pool size
        gpu_token_buffer_size = int(2 * (ubs * (avg_prompt_tokens + max_new_tokens)))
        cpu_token_pool_size = int(bs * (avg_prompt_tokens + max_new_tokens))
        token_to_kv_pool = TokenToKVPool(
            gpu_token_buffer_size,
            cpu_token_pool_size,
            dtype=torch.get_default_dtype(),
            head_num = model_config.num_key_value_heads // hardware_config.tp_size,
            head_dim = model_config.hidden_size // model_config.num_attention_heads,
            layer_num = model_config.num_hidden_layers,
        )

        # experts cache
        # todo change optimizer to output the number of experts on gpu as a policy parameter
        num_layers = model_config.num_hidden_layers
        num_comp_experts = policy.eg
        num_experts_gpu=int(model_config.num_local_experts * policy.wg)
        if num_comp_experts != model_config.num_local_experts:
            assert num_experts_gpu == 0
        # if we have enough GPU memory, the buffer is 2 * (num_experts - num_experts_gpu)
        # elif we do not have enough GPU memory, the buffer is of size 2 * num_comp_experts
        experts_pool_size  = num_layers * num_experts_gpu + 2 * (num_comp_experts - num_experts_gpu)
        experts_cache = torch.empty(experts_pool_size, 3 * model_config.intermediate_size * model_config.hidden_size, dtype=torch.get_default_dtype(), device="cuda")

        experts_pin = torch.empty(((num_comp_experts - num_experts_gpu), 
                                    3 * model_config.intermediate_size * model_config.hidden_size), 
                                    dtype=torch.get_default_dtype(), device="cpu").pin_memory()

        num_q_heads = model_config.num_attention_heads // hardware_config.tp_size
        n_kv_heads = model_config.num_key_value_heads // hardware_config.tp_size
        head_dim = model_config.hidden_size // model_config.num_attention_heads
        qkv_pin = [torch.empty(ubs, (num_q_heads + 2 * n_kv_heads) * head_dim, 
                              dtype=torch.get_default_dtype(), device="cpu").pin_memory() for _ in range(policy.n_ub)]

        hidden_pin = [torch.empty(ubs, 1, num_q_heads, head_dim,
                                 dtype=torch.get_default_dtype(), device="cpu").pin_memory() for _ in range(policy.n_ub)]

        #  gpu memory usage
        free_gpu_memory, _ = torch.cuda.mem_get_info()
        print(f"Free GPU memory: {free_gpu_memory / (1 << 30)}")


        # cpu workers
        cpu_executor = ThreadPoolExecutor(max_workers=1)
        copy_executor = ThreadPoolExecutor(max_workers=1)

        # cuda events and streams
        prefetch_stream = torch.cuda.Stream() 
        offload_stream = torch.cuda.Stream()
        load_stream = torch.cuda.Stream()
        cur_stream = torch.cuda.current_stream()
        offload_kv_events = [torch.cuda.Event() for _ in range(2)]
        


        return cls(model_config=model_config,
                   policy=policy,
                   token_to_kv_pool=token_to_kv_pool,
                   experts_cache=experts_cache,
                   qkv_pin=qkv_pin,
                   hidden_pin=hidden_pin,
                   experts_pin=experts_pin,
                   cpu_executor=cpu_executor,
                   copy_executor=copy_executor,
                   prefetch_stream=prefetch_stream,
                   offload_stream=offload_stream,
                   load_stream=load_stream,
                   cur_stream=cur_stream,
                   offload_kv_events=offload_kv_events,
                   avg_prompt_tokens=avg_prompt_tokens,
                   gen_len=max_new_tokens)

    def init_gpu_experts(self, cpu_experts_mem):
        num_gpu_experts = int(self.model_config.num_local_experts * self.policy.wg)
        if num_gpu_experts != 0:
            for i in range(self.model_config.num_hidden_layers):
                self.experts_cache[i * num_gpu_experts: (i + 1) * num_gpu_experts].copy_(cpu_experts_mem[i, :num_gpu_experts])
    
    def get_ecache_size(self):
        return self.experts_cache.shape[0]
    
    def delete_gpu_context(self):
        self.token_to_kv_pool.delete_gpu_cache()
        


        
    
