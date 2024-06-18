import asyncio
import logging
import multiprocessing
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List

import rpyc
import torch
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer
from fastmoe.utils.hf_transformers_utils import get_tokenizer
from fastmoe.serve.io_struct import (
    BatchTokenIDOut,
    BatchTokenizedGenerateReqInput,
)
from fastmoe.backend.execution_engine import ExecutionEngine
from fastmoe.backend.utils import HardwareConfig
from fastmoe.backend.task import Batch, Req
from fastmoe.backend.model_runner import ModelRunner, _set_default_torch_dtype
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
        self.int_token_logit_bias = torch.tensor(
            get_int_token_logit_bias(self.tokenizer, self.model_config.vocab_size)
        )
        set_random_seed(server_args.random_seed)
        logger.info(
            f"Rank {self.tp_rank}: "
            f"context_len={self.model_config.context_len}, "
        )

        self.hardware_config = HardwareConfig.init(torch.cuda.get_device_name(0), 
                                                   self.model_runner.total_cpu_memory,
                                                   server_args.cpu_mem_bdw,
                                                   server_args.tp_size
                                                   )

        # Init running status
        self.forward_queue: List[Req] = []
        self.running_batch: Batch = None
        self.out_pyobjs = []
        self.decode_forward_ct = 0
        self.stream_interval = server_args.stream_interval

        self.exe_engine: ExecutionEngine = None

        with _set_default_torch_dtype(torch.float16):
            self.build_tasks_and_exec_ctx(server_args.avg_prompt_len, server_args.gen_len)

    def flush_cache(self):
        if len(self.forward_queue) == 0 and (
            self.running_batch is None or len(self.running_batch.reqs) == 0
        ):
            self.exe_engine.reset()
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
                if isinstance(recv_req, BatchTokenizedGenerateReqInput):
                    self.handle_batch_generate_request(recv_req)
                    # Execute Graph
                    self.batch_forward_step()
                    # Clean up
                    self.flush_cache()
                else:
                    raise ValueError(f"Invalid request: {recv_req}")

        except Exception:
            logger.error("Exception in ModelRpcClient:\n" + get_exception_traceback())

        # Return results
        ret = self.out_pyobjs
        self.out_pyobjs = []
        return ret
    
    def build_tasks_and_exec_ctx(self, avg_prompt_len, gen_len):
        self.exe_engine = ExecutionEngine(self.model_runner, self.model_config, self.hardware_config, avg_prompt_len, gen_len)
        self.exe_engine.init_gpu_experts()
        torch.cuda.synchronize()

    @torch.inference_mode()
    def batch_forward_step(self):
        num_mb, abort_requests = self.exe_engine.create_micro_batches(self.forward_queue)
        self.forward_queue = []
        if abort_requests:
            self.handle_aborted_requests(abort_requests)
        self.exe_engine.init_weights_prefetch_meta()

        # prefill
        self.exe_engine.prepare_for_prefill(self.int_token_logit_bias)
        self.exe_engine.prefetch_experts(0)
        for i in range(self.model_config.num_hidden_layers):
            for j in range(self.exe_engine.weights_prefetch_num_pages_cpu):
                for k in range(num_mb):
                    # preattn and attn are done in layer(i, 0, k)
                    self.exe_engine.layer(i, j, k)
                    if j == 0:
                        self.exe_engine.offload_kv_cache(i, k)
                    if num_mb > 1:
                        self.exe_engine.offload_hidden_prefill(k)
                        self.exe_engine.load_hidden_prefill((k + 1) % num_mb)
                if j == self.exe_engine.weights_prefetch_num_pages_cpu - 1:
                    self.exe_engine.prefetch_experts((i + 1) % self.model_config.num_hidden_layers, 0, 0)
                else:
                    self.exe_engine.prefetch_experts(i, j + 1, 0)

        # update execution context for decode if necessary
        # decode
        decode_step = 0
        num_decode_steps = self.exe_engine.context.gen_len - 1 if num_mb > 1 else 1
        while decode_step < num_decode_steps:
            print("decode step: ", decode_step)
            self.exe_engine.prepare_for_decode()
            # Prologue
            for k, ub in enumerate(self.exe_engine.micro_batches[:2]):
                self.exe_engine.pre_attention(0, k)
                self.exe_engine.offload_qkv(0, k)
                self.exe_engine.cpu_attention(0, k)
                if self.exe_engine.weights_prefetch_num_pages_cpu > 1:
                    self.exe_engine.prefetch_experts_to_pin(0, k)
                else:
                    self.exe_engine.prefetch_experts_to_pin(1, k)
            
            for i in range(self.model_config.num_hidden_layers):
                j = 0
                for k in range(num_mb):
                    self.exe_engine.load_hidden(i, k)
                    if self.exe_engine.weights_prefetch_num_pages_cpu > 1:
                        # expert-granularity
                        self.exe_engine.prefetch_experts(i, 1, k, stage="decode")
                    else:
                        self.exe_engine.prefetch_experts((i + 1) % self.model_config.num_hidden_layers, 0, k, stage="decode")
                    self.exe_engine.post_attention(i, 0, k)
                    if num_mb > 2:
                        if k + 2 < num_mb:
                            self.exe_engine.pre_attention(i, k + 2)
                            self.exe_engine.offload_qkv(i, k + 2)
                            self.exe_engine.cpu_attention(i, k + 2)
                        elif k + 2 >= num_mb and i < self.model_config.num_hidden_layers - 1:
                            self.exe_engine.pre_attention(i + 1, (k + 2) % num_mb)
                            self.exe_engine.offload_qkv(i + 1, (k + 2) % num_mb)
                            self.exe_engine.cpu_attention(i + 1, (k + 2) % num_mb)
                    
                    layer_id, slot_id = self.exe_engine.get_prefetch_e2p_idx(i, j, k, num_mb)
                    if layer_id is not None and slot_id is not None:
                        self.exe_engine.prefetch_experts_to_pin(layer_id, slot_id)
                # self.exe_engine.prefetch_experts((i + 1) % self.model_config.num_hidden_layers, 0, 0)
                    
                    
                # for expert-granularity compute
                for j in range(1, self.exe_engine.weights_prefetch_num_pages_cpu):
                    for k in range(num_mb):
                        if j == self.exe_engine.weights_prefetch_num_pages_cpu - 1:
                            self.exe_engine.prefetch_experts((i + 1) % self.model_config.num_hidden_layers, 0, k, stage="decode")
                        else:
                            self.exe_engine.prefetch_experts(i, j, k, stage="decode")
                        self.exe_engine.post_attention(i, j, k)
                        
                        layer_id, slot_id = self.exe_engine.get_prefetch_e2p_idx(i, j, k, num_mb)
                        if layer_id and slot_id:
                            self.exe_engine.prefetch_experts_to_pin(layer_id, slot_id)
                            
            decode_step += 1
        
        for ub in self.exe_engine.micro_batches:
            self.handle_finished_requests(ub)
    
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
            )
            self.forward_queue.append(req)

    def handle_finished_requests(self, batch: Batch):
        output_rids = []
        output_tokens = []
        output_hit_stop_str = []
        output_skip_special_tokens = []
        output_meta_info = []
        output_finished = []
        finished_indices = []
        for i, req in enumerate(batch.reqs):
            if req.finished:
                finished_indices.append(i)

            if req.finished:
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

        # Clear Batch
        if finished_indices:
            batch.reqs = []
    
    def handle_aborted_requests(self, requests: List[Req]):
        output_rids = []
        output_tokens = []
        output_hit_stop_str = []
        output_skip_special_tokens = []
        output_meta_info = []
        output_finished = []
        for req in requests:
            if req.finished:
                output_rids.append(req.rid)
                output_tokens.append([])
                output_hit_stop_str.append(req.hit_stop_str)
                output_skip_special_tokens.append(
                    req.sampling_params.skip_special_tokens
                )
                meta_info = {
                    "prompt_tokens": len(req.input_ids),
                    "completion_tokens": 0,
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
