import asyncio
import logging
import multiprocessing
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import rpyc
import torch
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer
from fastmoe.utils.hf_transformers_utils import get_tokenizer
from fastmoe.serve.io_struct import (
    BatchTokenIDOut,
    FlushCacheReq,
    TokenizedGenerateReqInput,
    BatchTokenizedGenerateReqInput,
)
from fastmoe.serve.router.infer_batch import Batch, ForwardMode, Req, DecodePart, KVLayout
from fastmoe.serve.router.model_runner import ModelRunner
from fastmoe.serve.router.scheduler import Scheduler
from fastmoe.serve.server_args import PortArgs, ServerArgs
from fastmoe.utils.model_config import ModelConfig
from fastmoe.utils.utils import (
    get_exception_traceback,
    get_int_token_logit_bias,
    set_random_seed,
)

logger = logging.getLogger("model_rpc")


class ModelRpcServer(rpyc.Service):
    def exposed_init_model(
        self,
        tp_rank: int,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        server_args, port_args = [obtain(x) for x in [server_args, port_args]]

        # Copy arguments
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size

        # Init model and tokenizer
        self.model_config = ModelConfig(
            server_args.model_path, server_args.trust_remote_code
        )
        self.model_runner = ModelRunner(
            self.model_config,
            server_args.mem_fraction_static,
            tp_rank,
            server_args.tp_size,
            port_args.nccl_port,
            server_args.load_format,
            server_args.trust_remote_code,
        )
        
        self.tokenizer = get_tokenizer(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
        )
        self.eos_token_id = self.tokenizer.eos_token_id
        self.max_total_num_token = self.model_runner.max_total_num_token
        self.max_num_running_seq = self.max_total_num_token // 2
        self.max_prefill_num_token = max(
            self.model_config.context_len,
            self.max_total_num_token // 6
            if server_args.max_prefill_num_token is None
            else server_args.max_prefill_num_token,
        )
        self.int_token_logit_bias = torch.tensor(
            get_int_token_logit_bias(self.tokenizer, self.model_config.vocab_size)
        )
        set_random_seed(server_args.random_seed)
        logger.info(
            f"Rank {self.tp_rank}: "
            f"max_total_num_token={self.max_total_num_token}, "
            f"max_prefill_num_token={self.max_prefill_num_token}, "
            f"context_len={self.model_config.context_len}, "
        )
        
        self.scheduler = Scheduler(
            self.max_num_running_seq,
            self.max_prefill_num_token,
            self.max_total_num_token,
        )
        self.req_to_token_pool = self.model_runner.req_to_token_pool
        self.req_to_cpu_token_pool = self.model_runner.req_to_cpu_token_pool
        self.token_to_kv_pool = self.model_runner.token_to_kv_pool

        # Init running status
        self.forward_queue: List[Req] = []
        self.running_batch: Batch = None
        self.out_pyobjs = []
        self.decode_forward_ct = 0
        self.stream_interval = server_args.stream_interval

        # Init new token estimation
        self.new_token_ratio = min(0.4 * server_args.schedule_conservativeness, 1.0)
        self.min_new_token_ratio = min(0.2 * server_args.schedule_conservativeness, 1.0)
        self.new_token_ratio_step = (0.0001, 0.05)  # (down, up)

        # todo @caoshiyi, profile
        self.max_bs = 400
        self.max_new_tokens = 40000
        self.max_output_len = 32

    def flush_cache(self):
        if len(self.forward_queue) == 0 and (
            self.running_batch is None or len(self.running_batch.reqs) == 0
        ):
            self.req_to_token_pool.clear()
            self.req_to_cpu_token_pool.clear()
            self.token_to_kv_pool.clear()
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
        else:
            warnings.warn(
                "Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.forward_queue)}, "
                f"#running-req: {0 if self.running_batch is None else len(self.running_batch.reqs)}"
            )

    def exposed_step(self, recv_reqs):
        if self.tp_size != 1:
            recv_reqs = obtain(recv_reqs)

        try:
            # Recv requests
            for recv_req in recv_reqs:
                if isinstance(recv_req, TokenizedGenerateReqInput):
                    self.handle_generate_request(recv_req)
                elif isinstance(recv_req, BatchTokenizedGenerateReqInput):
                    self.handle_batch_generate_request(recv_req)
                    self.batch_forward_step()
                    self.flush_cache()
                elif isinstance(recv_req, FlushCacheReq):
                    self.flush_cache()
                else:
                    raise ValueError(f"Invalid request: {recv_req}")

            # Forward
            # self.forward_step()
        except Exception:
            logger.error("Exception in ModelRpcClient:\n" + get_exception_traceback())

        # Return results
        ret = self.out_pyobjs
        self.out_pyobjs = []
        return ret
    
    @torch.inference_mode()
    def batch_forward_step(self):
        num_buffers = 2
        micro_batches = self.generate_micro_batches()
        print("Num Micro Batches:", len(micro_batches))
        # workers and streams
        cpu_executor = ThreadPoolExecutor(max_workers=1)
        copy_executor = ThreadPoolExecutor(max_workers=1)
        prefetch_stream = torch.cuda.Stream()
        offload_stream = torch.cuda.Stream()
        load_stream = torch.cuda.Stream()
        cur_stream = torch.cuda.current_stream()
        prefetch_events = [torch.cuda.Event() for _ in range(len(micro_batches))]
        copy_futures = [None for _ in range(len(micro_batches))]

        # prefill
        step_num_layers, prefetch_experts, new_micro_batches = self.shuffle_micro_batches(micro_batches)
        with torch.cuda.stream(prefetch_stream):
            self.model_runner.model.prefetch_expert(prefetch_experts[0])
        torch.cuda.synchronize()
        while micro_batches[0].cur_layer < self.model_config.num_hidden_layers:
            prefetch_layer = micro_batches[0].cur_layer + 1 if micro_batches[0].cur_layer < self.model_config.num_hidden_layers - 1 else 0
            for i, micro_batch in enumerate(new_micro_batches):
                self.forward_partial_fill_batch(micro_batch, micro_batch.cur_layer, step_num_layers, i%num_buffers)
                # with torch.cuda.stream(load_stream):
                #     if new_micro_batches[(i+1)% len(new_micro_batches)].hidden_states is not None:
                #         new_micro_batches[(i+1)% len(new_micro_batches)].hidden_states = new_micro_batches[(i+1)% len(new_micro_batches)].hidden_states.to('cuda', non_blocking=True)
                with torch.cuda.stream(offload_stream):
                    micro_batch.attn_event.wait(offload_stream)
                    micro_batch.offload_kv_cache()
                    # if micro_batch.hidden_states is not None:
                    #     micro_batch.hidden_states = micro_batch.hidden_states.to('cpu', non_blocking=True)
                micro_batch.cur_layer += step_num_layers
            with torch.cuda.stream(prefetch_stream):
                self.model_runner.model.prefetch_expert(prefetch_experts[prefetch_layer])
            
        # decode
        # total_partitions = 1 if len(new_micro_batches) <= len(prefetch_experts[0]) else math.ceil(len(new_micro_batches) / len(prefetch_experts[0]))
        total_partitions = 1
        decode_step = 0
        while True:
            new_micro_batches = []
            for micro_batch in micro_batches:
                if micro_batch.is_empty():
                    continue
                # reset states:
                micro_batch.cur_layer = 0
                micro_batch.hidden_states = None
                micro_batch.residual = None
                new_micro_batches.append(micro_batch)
            if len(new_micro_batches) == 0:
                break
            micro_batches = new_micro_batches
            print("decode_step:", decode_step)
            for micro_batch in new_micro_batches:
                # Build batch tensors
                micro_batch.prepare_for_partial_decode()
            
            # Prologue
            for i, micro_batch in enumerate(new_micro_batches[:2]):
                # PREATTN and CPU_ATTN for [0, 0] and [1, 0]
                self.forward_partial_decode_batch(micro_batch, step_num_layers, DecodePart.PREATTN, micro_batch.cur_layer)
                micro_batch.compute_event.record()
                with torch.cuda.stream(offload_stream):
                    micro_batch.compute_event.wait(offload_stream)
                    micro_batch.qkv_pin.copy_(micro_batch.qkv, non_blocking=True)
                    micro_batch.offload_event.record(offload_stream)
                micro_batch.attn_future = cpu_executor.submit(self.forward_partial_decode_batch, micro_batch, step_num_layers, DecodePart.CPU_ATTN, micro_batch.cur_layer)
                # prefetch W[1, i] to pin
                copy_futures[i] = copy_executor.submit(self.model_runner.model.prefetch_experts_to_pin, 1, 
                                                        i // total_partitions, 
                                                        i // len(prefetch_experts[0]), 
                                                        total_partitions, 
                                                        prefetch_events[i])

            while micro_batches[0].cur_layer < self.model_config.num_hidden_layers:
                prefetch_events[-1].wait(cur_stream)
                prefetch_layer = micro_batches[0].cur_layer + 1 if micro_batches[0].cur_layer < self.model_config.num_hidden_layers - 1 else 0
                
                for i, micro_batch in enumerate(new_micro_batches):
                    micro_batch.attn_future.result()

                    # POSTATTN for [i, j]
                    with torch.cuda.stream(load_stream):
                        micro_batch.hidden_states = micro_batch.hidden_pin.to('cuda', non_blocking=True)
                        micro_batch.load_event.record(load_stream)
                        if i < len(prefetch_experts[prefetch_layer]) * total_partitions:
                            copy_futures[i].result()
                            self.model_runner.model.prefetch_experts_from_pin(prefetch_layer, 
                                                                              i // total_partitions, 
                                                                              i // len(prefetch_experts[0]), 
                                                                              total_partitions)
                            prefetch_events[i].record(load_stream)
                    
                    micro_batch.load_event.wait(cur_stream)
                    self.forward_partial_decode_batch(micro_batch, step_num_layers, DecodePart.POSTATTN, micro_batch.cur_layer)
                    
                    if i + 2 < len(new_micro_batches):
                        # PREATTN and CPU_ATTN for [i+2, j] (j < num_layers)
                        pre_micro_batch = new_micro_batches[i+2]
                        self.forward_partial_decode_batch(pre_micro_batch, step_num_layers, DecodePart.PREATTN, micro_batch.cur_layer)
                        pre_micro_batch.compute_event.record()
                        with torch.cuda.stream(offload_stream):
                            pre_micro_batch.compute_event.wait(offload_stream)
                            pre_micro_batch.qkv_pin.copy_(pre_micro_batch.qkv, non_blocking=True)
                            pre_micro_batch.offload_event.record(offload_stream)
                        pre_micro_batch.attn_future = cpu_executor.submit(self.forward_partial_decode_batch, pre_micro_batch, step_num_layers, DecodePart.CPU_ATTN, micro_batch.cur_layer)
                        # prefetch weights to pin
                        if i + 2 < len(prefetch_experts[prefetch_layer]) * total_partitions:
                            copy_futures[i + 2] = copy_executor.submit(self.model_runner.model.prefetch_experts_to_pin, 
                                                                       prefetch_layer,  
                                                                        (i + 2) // total_partitions,
                                                                        (i + 2) // len(prefetch_experts[0]),
                                                                        total_partitions, 
                                                                        prefetch_events[i + 2])
                    elif i + 2 >= len(new_micro_batches) and micro_batch.cur_layer < self.model_config.num_hidden_layers - 1:
                        # PREATTN for [(i+2)%B, j+1] (j < num_layers - 1)
                        pre_micro_batch = new_micro_batches[(i+2)%len(new_micro_batches)]
                        assert pre_micro_batch.cur_layer < self.model_config.num_hidden_layers
                        self.forward_partial_decode_batch(pre_micro_batch, step_num_layers, DecodePart.PREATTN, micro_batch.cur_layer + 1)
                        pre_micro_batch.compute_event.record()
                        with torch.cuda.stream(offload_stream):
                            pre_micro_batch.compute_event.wait(offload_stream)
                            pre_micro_batch.qkv_pin.copy_(pre_micro_batch.qkv, non_blocking=True)
                            pre_micro_batch.offload_event.record(offload_stream)
                        pre_micro_batch.attn_future = cpu_executor.submit(self.forward_partial_decode_batch, pre_micro_batch, step_num_layers, DecodePart.CPU_ATTN, micro_batch.cur_layer + 1)
                        # prefetch weights to pin
                        copy_futures[(i + 2)%len(new_micro_batches)] = copy_executor.submit(self.model_runner.model.prefetch_experts_to_pin, 
                                                                                            (prefetch_layer + 1) % self.model_config.num_hidden_layers, 
                                                                                            (i + 2)%len(new_micro_batches) // total_partitions, 
                                                                                            (i + 2)%len(new_micro_batches) // len(prefetch_experts[0]), 
                                                                                            total_partitions, 
                                                                                            prefetch_events[(i + 2)%len(new_micro_batches)])
                    
                    micro_batch.cur_layer += step_num_layers
                # with torch.cuda.stream(prefetch_stream):
                #     self.model_runner.model.prefetch_expert(prefetch_experts[prefetch_layer])
            decode_step += 1

    def shuffle_micro_batches(self, micro_batches):
        # todo @caoshiyi, advanced schedule
        num_hidden_layers = self.model_config.num_hidden_layers
        num_experts = self.model_config.num_local_experts
        prefetch_experts = [[(next_layer + i, j) for j in range(num_experts) for i in range(1)] for next_layer in range(num_hidden_layers)]
        return 1, prefetch_experts, micro_batches

    # todo @caoshiyi, init_step_num_layers as a parameter
    def generate_micro_batches(self, init_step_num_layers=1):
        micro_batches = []
        cpu_available_size = self.token_to_kv_pool.cpu_available_size()
        while cpu_available_size > 0:
            # Add requests if there is available space
            can_run_list = []
            new_batch_total_tokens = 0
            new_batch_input_tokens = 0

            available_size = self.token_to_kv_pool.available_size()

            for req in self.forward_queue:
                # todo
                if (
                    (req.input_len + req.max_new_tokens() + new_batch_total_tokens) * init_step_num_layers
                    < available_size
                    and req.input_len + new_batch_input_tokens
                    < self.max_prefill_num_token and len(can_run_list) < 400
                ):

                    can_run_list.append(req)
                    new_batch_total_tokens += (
                        req.input_len + req.max_new_tokens()
                    )
                    new_batch_input_tokens += req.input_len

            if len(can_run_list) == 0:
                break

            new_batch = Batch.init_new(
                can_run_list,
                self.req_to_token_pool,
                self.token_to_kv_pool,
                self.req_to_cpu_token_pool,
            )
            self.forward_queue = [x for x in self.forward_queue if x not in can_run_list]
            cpu_available_size -= new_batch_total_tokens
            micro_batches.append(new_batch)
            print("Append new micro batch. #reqs:", len(new_batch.reqs), "new_batch_input_tokens:", new_batch_input_tokens)
        return micro_batches
    
    def forward_partial_fill_batch(self, batch: Batch, cur_layer, step_num_layers, buffer_idx=0):
        if cur_layer == 0:
            # Build batch tensors
            batch.prepare_for_partial(
                self.model_config.vocab_size, self.int_token_logit_bias, self.model_config.num_hidden_layers,
                self.max_bs, self.max_new_tokens, self.max_output_len, buffer_idx, kvlayout=KVLayout.Continuous, num_q_heads=self.model_config.num_attention_heads
            )

        if batch.new_num_tokens != 0:
            if cur_layer + step_num_layers != self.model_config.num_hidden_layers:
                print("cur_layer:", cur_layer, "step_num_layers:", step_num_layers)
                # partial forward
                batch.hidden_states, batch.residual = self.model_runner.forward(
                    batch, ForwardMode.PARTIAL, DecodePart.ALL, cur_layer, batch.return_logprob
                )
            else:
                print("cur_layer:", cur_layer, "step_num_layers:", step_num_layers)
                # last layer forward
                logits, (logprobs, normalized_logprobs) = self.model_runner.forward(
                    batch, ForwardMode.PARTIAL, DecodePart.ALL, cur_layer, batch.return_logprob
                )
                if logprobs is not None:
                    logprobs = logprobs.cpu().tolist()
                    normalized_logprobs = normalized_logprobs.cpu().tolist()

                next_token_ids, next_token_probs = batch.sample(logits)
                next_token_ids = next_token_ids.cpu().tolist()
                batch.hidden_states = None
                batch.residual = None
        else:
            next_token_ids = [self.tokenizer.eos_token_id] * len(batch.reqs)
            logprobs = normalized_logprobs = None

        if cur_layer + step_num_layers == self.model_config.num_hidden_layers:
            # Check finish condition
            reqs = batch.reqs
            pt = 0
            for i, req in enumerate(reqs):
                req.output_ids = [next_token_ids[i]]
                req.check_finished()

                if logprobs is not None:
                    req.logprob = logprobs[pt : pt + req.input_len - 1]
                    req.normalized_logprob = normalized_logprobs[i]
                    pt += req.input_len

            self.handle_finished_requests(batch, ForwardMode.PARTIAL)

    def forward_partial_decode_batch(self, batch: Batch, step_num_layers, decode_part: DecodePart, cur_layer: int):
        print("cur_layer:", cur_layer, "step_num_layers:", step_num_layers, "decode_part:", decode_part)
        
        if decode_part == DecodePart.CPU_ATTN:
            batch.offload_event.synchronize()
            return self.model_runner.forward(
                batch, ForwardMode.PARTIAL_DECODE, decode_part, cur_layer
            )
        elif decode_part == DecodePart.PREATTN:
            batch.qkv, batch.residual = self.model_runner.forward(
                batch, ForwardMode.PARTIAL_DECODE, decode_part, cur_layer
            )
        elif cur_layer + step_num_layers != self.model_config.num_hidden_layers:
            # partial decode
            batch.hidden_states, batch.residual = self.model_runner.forward(
                batch, ForwardMode.PARTIAL_DECODE, decode_part, cur_layer
            )
        else:
            # last layer forward
            logits, _ = self.model_runner.forward(batch, ForwardMode.PARTIAL_DECODE, decode_part, cur_layer)
            next_token_ids, next_token_probs = batch.sample(logits)
            next_token_ids = next_token_ids.cpu().tolist()

        if cur_layer + step_num_layers == self.model_config.num_hidden_layers :
            if decode_part == DecodePart.ALL or decode_part == DecodePart.POSTATTN:
                # Check finish condition
                reqs = batch.reqs
                for i in range(len(reqs)):
                    reqs[i].output_ids.append(next_token_ids[i])
                    reqs[i].check_finished()

                self.handle_finished_requests(batch, ForwardMode.PARTIAL_DECODE)
    
    @torch.inference_mode()
    def forward_step(self):
        new_batch = self.get_new_fill_batch()

        if new_batch is not None:
            # Run new fill batch
            self.forward_fill_batch(new_batch)

            if not new_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = new_batch
                else:
                    self.running_batch.merge(new_batch)
        else:
            # Run decode batch
            if self.running_batch is not None:
                # Run a few decode batches continuously for reducing overhead
                for _ in range(10):
                    self.forward_decode_batch(self.running_batch)

                    if self.running_batch.is_empty():
                        self.running_batch = None
                        break

                    if self.out_pyobjs and self.running_batch.reqs[0].stream:
                        break
            else:
                # check the available size
                available_size =self.token_to_kv_pool.available_size()
                if available_size != self.max_total_num_token:
                    warnings.warn(
                        "Warning: "
                        f"available_size={available_size}, max_total_num_token={self.max_total_num_token}\n"
                        "KV cache pool leak detected!"
                    )

        if self.running_batch is not None and self.tp_rank == 0:
            if self.decode_forward_ct % 20 == 0:
                num_used = self.max_total_num_token - self.token_to_kv_pool.available_size()
                logger.info(
                    f"#running-req: {len(self.running_batch.reqs)}, "
                    f"#token: {num_used}, "
                    f"token usage: {num_used / self.max_total_num_token:.2f}, "
                    f"#queue-req: {len(self.forward_queue)}"
                )

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        req = Req(recv_req.rid, recv_req.input_text, recv_req.input_ids)
        req.sampling_params = recv_req.sampling_params
        req.return_logprob = recv_req.return_logprob
        req.logprob_start_len = recv_req.logprob_start_len
        req.stream = recv_req.stream
        req.tokenizer = self.tokenizer

        # Truncate long prompts
        req.input_ids = req.input_ids[: self.model_config.context_len - 1]
        req.sampling_params.max_new_tokens = min(
            req.sampling_params.max_new_tokens,
            self.model_config.context_len - 1 - len(req.input_ids),
            self.max_total_num_token - 128 - len(req.input_ids),
        )
        self.forward_queue.append(req)
    
    def handle_batch_generate_request(
        self,
        recv_req: BatchTokenizedGenerateReqInput,
    ):
        for i in range(len(recv_req.rid)):
            req = Req(recv_req.rid[i], recv_req.input_text[i], recv_req.input_ids[i])
            req.sampling_params = recv_req.sampling_params
            req.return_logprob = recv_req.return_logprob
            req.logprob_start_len = recv_req.logprob_start_len
            req.stream = recv_req.stream
            req.tokenizer = self.tokenizer

            # Truncate long prompts
            req.input_ids = req.input_ids[: self.model_config.context_len - 1]
            req.sampling_params.max_new_tokens = min(
                req.sampling_params.max_new_tokens,
                self.model_config.context_len - 1 - len(req.input_ids),
                self.max_total_num_token - 128 - len(req.input_ids),
            )
            self.forward_queue.append(req)

    def get_new_fill_batch(self):
        if (
            self.running_batch is not None
            and len(self.running_batch.reqs) > self.max_num_running_seq
        ):
            return None

        # Get priority queue
        self.forward_queue = self.scheduler.get_priority_queue(self.forward_queue)

        # Add requests if there is available space
        can_run_list = []
        new_batch_total_tokens = 0
        new_batch_input_tokens = 0

        available_size = self.token_to_kv_pool.available_size()
        if self.running_batch:
            available_size -= sum(
                [
                    (r.max_new_tokens() - len(r.output_ids)) * self.new_token_ratio
                    for r in self.running_batch.reqs
                ]
            )

        for req in self.forward_queue:
            if (
                req.input_len + req.max_new_tokens() + new_batch_total_tokens
                < available_size
                and req.input_len + new_batch_input_tokens
                < self.max_prefill_num_token
            ):

                can_run_list.append(req)
                new_batch_total_tokens += (
                    req.input_len + req.max_new_tokens()
                )
                new_batch_input_tokens += req.input_len

        if len(can_run_list) == 0:
            return None

        if self.tp_rank == 0:
            running_req = (
                0 if self.running_batch is None else len(self.running_batch.reqs)
            )
            logger.info(
                f"new fill batch. #seq: {len(can_run_list)}. "
                f"#new_token: {new_batch_input_tokens}. "
                f"#remaining_req: {len(self.forward_queue) - len(can_run_list)}. "
                f"#running_req: {running_req}. "
            )

        new_batch = Batch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool,
        )
        self.forward_queue = [x for x in self.forward_queue if x not in can_run_list]
        return new_batch

    def forward_fill_batch(self, batch: Batch):
        # Build batch tensors
        batch.prepare_for_extend(
            self.model_config.vocab_size, self.int_token_logit_bias
        )

        if batch.new_num_tokens != 0:
            # Forward
            logits, (logprobs, normalized_logprobs) = self.model_runner.forward(
                batch, ForwardMode.PREFILL, batch.return_logprob
            )
            if logprobs is not None:
                logprobs = logprobs.cpu().tolist()
                normalized_logprobs = normalized_logprobs.cpu().tolist()

            next_token_ids, next_token_probs = batch.sample(logits)
            next_token_ids = next_token_ids.cpu().tolist()
        else:
            next_token_ids = [self.tokenizer.eos_token_id] * len(batch.reqs)
            logprobs = normalized_logprobs = None

        # Check finish condition
        reqs = batch.reqs
        pt = 0
        for i, req in enumerate(reqs):
            req.output_ids = [next_token_ids[i]]
            req.check_finished()

            if logprobs is not None:
                req.logprob = logprobs[pt : pt + req.input_len - 1]
                req.normalized_logprob = normalized_logprobs[i]
                pt += req.input_len

        self.handle_finished_requests(batch)

    def forward_decode_batch(self, batch: Batch):
        # check if decode out of memory
        if not batch.check_decode_mem():
            old_ratio = self.new_token_ratio
            self.new_token_ratio = min(old_ratio + self.new_token_ratio_step[1], 1.0)

            retracted_reqs = batch.retract_decode()
            logger.info(
                "decode out of memory happened, "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )
            self.forward_queue.extend(retracted_reqs)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_step[0],
                self.min_new_token_ratio,
            )

        # Update batch tensors
        self.decode_forward_ct = (self.decode_forward_ct + 1) % (1 << 30)
        batch.prepare_for_decode()

        # Forward
        logits = self.model_runner.forward(batch, ForwardMode.DECODE)
        next_token_ids, next_token_probs = batch.sample(logits)
        next_token_ids = next_token_ids.cpu().tolist()

        # Check finish condition
        reqs = batch.reqs
        for i in range(len(reqs)):
            reqs[i].output_ids.append(next_token_ids[i])
            reqs[i].check_finished()

        self.handle_finished_requests(batch)

    def handle_finished_requests(self, batch: Batch, forward_mode: ForwardMode = None):
        output_rids = []
        output_tokens = []
        output_hit_stop_str = []
        output_skip_special_tokens = []
        output_meta_info = []
        output_finished = []
        finished_indices = []
        unfinished_indices = []
        for i, req in enumerate(batch.reqs):
            if req.finished:
                finished_indices.append(i)
            else:
                unfinished_indices.append(i)

            if req.finished or (
                (
                    req.stream
                    and (
                        self.decode_forward_ct % self.stream_interval == 0
                        or len(req.output_ids) == 1
                    )
                )
            ):
                output_rids.append(req.rid)
                output_tokens.append(req.output_ids)
                output_hit_stop_str.append(req.hit_stop_str)
                output_skip_special_tokens.append(
                    req.sampling_params.skip_special_tokens
                )
                meta_info = {
                    "prompt_tokens": len(req.input_ids),
                    "completion_tokens": len(req.output_ids),
                }
                if req.return_logprob:
                    meta_info["prompt_logprob"] = req.logprob
                    meta_info["normalized_prompt_logprob"] = req.normalized_logprob
                output_meta_info.append(meta_info)
                output_finished.append(req.finished)

        # Send to detokenizer
        if output_rids:
            self.out_pyobjs.append(
                BatchTokenIDOut(
                    output_rids,
                    output_tokens,
                    output_hit_stop_str,
                    output_skip_special_tokens,
                    output_meta_info,
                    output_finished,
                )
            )

        # Remove finished reqs
        if finished_indices:
            # req_pool_indices_cpu = batch.req_pool_indices.cpu().tolist()
            # req_cpu_pool_indices_cpu = batch.req_cpu_pool_indices.cpu().tolist()
            # for i in finished_indices:
            #     req = batch.reqs[i]
            #     req_pool_idx = req_pool_indices_cpu[i]
            #     # req_cpu_pool_idx = req_cpu_pool_indices_cpu[i]
            #     token_ids = tuple(req.input_ids + req.output_ids)
            #     seq_len = len(token_ids) - 1
                # indices = self.req_to_token_pool.req_to_token[req_pool_idx, :seq_len]

                # self.token_to_kv_pool.free(indices.flatten())
                # self.req_to_token_pool.free(req_pool_idx)
                # self.req_to_cpu_token_pool.free(req_cpu_pool_idx)

            # Update batch tensors
            if unfinished_indices:
                batch.filter_batch(unfinished_indices, forward_mode)
            else:
                batch.reqs = []


class ModelRpcClient:
    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        tp_size = server_args.tp_size

        if tp_size == 1:
            # Init model
            self.model_server = ModelRpcServer()
            self.model_server.exposed_init_model(0, server_args, port_args)

            # Wrap functions
            def async_wrap(f):
                async def _func(*args, **kwargs):
                    return f(*args, **kwargs)

                return _func

            self.step = async_wrap(self.model_server.exposed_step)
        else:
            with ThreadPoolExecutor(tp_size) as executor:
                # Launch model processes
                rets = executor.map(start_model_process, port_args.model_rpc_ports)
                self.model_servers = [x[0] for x in rets]
                self.procs = [x[1] for x in rets]

                # Init model
                def init_model(i):
                    return self.model_servers[i].init_model(i, server_args, port_args)

                rets = [obtain(x) for x in executor.map(init_model, range(tp_size))]

            # Wrap functions
            def async_wrap(func_name):
                fs = [rpyc.async_(getattr(m, func_name)) for m in self.model_servers]

                async def _func(*args, **kwargs):
                    tasks = [f(*args, **kwargs) for f in fs]
                    await asyncio.gather(*[asyncio.to_thread(t.wait) for t in tasks])
                    return obtain(tasks[0].value)

                return _func

            self.step = async_wrap("step")


def start_model_process(port):
    def _init_service(port):
        t = ThreadedServer(
            ModelRpcServer(),
            port=port,
            protocol_config={"allow_pickle": True, "sync_request_timeout": 1800},
        )
        t.start()

    proc = multiprocessing.Process(target=_init_service, args=(port,))
    proc.start()
    time.sleep(1)

    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect(
                "localhost",
                port,
                config={"allow_pickle": True, "sync_request_timeout": 1800},
            )
            break
        except ConnectionRefusedError:
            time.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise RuntimeError("init rpc env error!")

    assert proc.is_alive()
    return con.root, proc
