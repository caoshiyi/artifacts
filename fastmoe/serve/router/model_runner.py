import importlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastmoe.serve.router.infer_batch import Batch, ForwardMode
from fastmoe.managers.memory import ReqToTokenPool, TokenToKVPool
from fastmoe.utils.utils import get_available_gpu_memory
from vllm.model_executor.model_loader import _set_default_torch_dtype
from vllm.model_executor.parallel_utils.parallel_state import initialize_model_parallel

import sglang

logger = logging.getLogger("model_runner")


# for model_mode
global_model_mode: List[str] = []


@lru_cache()
def import_model_classes():
    model_arch_name_to_cls = {}
    for module_path in (Path(sglang.__file__).parent / "srt" / "models").glob("*.py"):
        module = importlib.import_module(f"sglang.srt.models.{module_path.stem}")
        if hasattr(module, "EntryClass"):
            model_arch_name_to_cls[module.EntryClass.__name__] = module.EntryClass
    return model_arch_name_to_cls


def get_model_cls_by_arch_name(model_arch_names):
    model_arch_name_to_cls = import_model_classes()

    model_class = None
    for arch in model_arch_names:
        if arch in model_arch_name_to_cls:
            model_class = model_arch_name_to_cls[arch]
            break
    else:
        raise ValueError(
            f"Unsupported architectures: {arch}. "
            f"Supported list: {list(model_arch_name_to_cls.keys())}"
        )
    return model_class


@dataclass
class InputMetadata:
    model_runner: "ModelRunner"
    forward_mode: ForwardMode
    batch_size: int
    total_num_tokens: int
    max_seq_len: int
    req_pool_indices: torch.Tensor
    start_loc: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: TokenToKVPool

    out_cache_loc: torch.Tensor = None
    out_cache_cont_start: torch.Tensor = None
    out_cache_cont_end: torch.Tensor = None

    other_kv_index: torch.Tensor = None
    return_logprob: bool = False

    @classmethod
    def create(
        cls,
        model_runner,
        tp_size,
        forward_mode,
        req_pool_indices,
        seq_lens,
        position_ids_offsets,
        out_cache_loc,
        out_cache_cont_start=None,
        out_cache_cont_end=None,
        return_logprob=False,
    ):
        batch_size = len(req_pool_indices)
        start_loc = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
        start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)
        total_num_tokens = int(torch.sum(seq_lens))
        max_seq_len = int(torch.max(seq_lens))

        if forward_mode == ForwardMode.DECODE:
            positions = ((seq_lens - 1) + position_ids_offsets).to(torch.int64)
            other_kv_index = model_runner.req_to_token_pool.req_to_token[
                req_pool_indices[0], seq_lens[0] - 1
            ].item()
        else:
            seq_lens_np = seq_lens.cpu().numpy()
            position_ids_offsets_np = position_ids_offsets.cpu().numpy()
            positions = torch.tensor(
                np.concatenate(
                    [
                        np.arange(
                            position_ids_offsets_np[i],
                            seq_lens_np[i] + position_ids_offsets_np[i],
                        )
                        for i in range(batch_size)
                    ],
                    axis=0,
                ),
                device="cuda",
            )
            other_kv_index = None

        ret = cls(
            model_runner=model_runner,
            forward_mode=forward_mode,
            batch_size=batch_size,
            total_num_tokens=total_num_tokens,
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            start_loc=start_loc,
            seq_lens=seq_lens,
            positions=positions,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            out_cache_cont_start=out_cache_cont_start,
            out_cache_cont_end=out_cache_cont_end,
            return_logprob=return_logprob,
            other_kv_index=other_kv_index,
        )

        return ret


class ModelRunner:
    def __init__(
        self,
        model_config,
        mem_fraction_static,
        tp_rank,
        tp_size,
        nccl_port,
        load_format="auto",
        trust_remote_code=True,
        model_mode: List[str] = (),
    ):
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.nccl_port = nccl_port
        self.load_format = load_format
        self.trust_remote_code = trust_remote_code
        self.model_mode = model_mode

        global global_model_mode
        global_model_mode = model_mode

        # Init torch distributed
        torch.cuda.set_device(self.tp_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=self.tp_size,
            rank=self.tp_rank,
            init_method=f"tcp://127.0.0.1:{self.nccl_port}",
        )

        # A small all_reduce for warmup.
        if self.tp_size > 1:
            torch.distributed.all_reduce(torch.zeros(1).cuda())
        initialize_model_parallel(tensor_model_parallel_size=self.tp_size)

        total_gpu_memory = get_available_gpu_memory(
            self.tp_rank, distributed=self.tp_size > 1
        ) * (1 << 30)
        self.load_model()
        self.init_memory_pool(total_gpu_memory)

    def load_model(self):
        """See also vllm/model_executor/model_loader.py::get_model"""
        # Select model class
        architectures = getattr(self.model_config.hf_config, "architectures", [])
        model_class = get_model_cls_by_arch_name(architectures)
        logger.info(f"Rank {self.tp_rank}: load weight begin.")

        # Load weights
        linear_method = None
        with _set_default_torch_dtype(torch.float16):
            with torch.device("cuda"):
                model = model_class(
                    config=self.model_config.hf_config, linear_method=linear_method
                )
            model.load_weights(
                self.model_config.path,
                cache_dir=None,
                load_format=self.load_format,
                revision=None,
            )
        self.model = model.eval()

        logger.info(f"Rank {self.tp_rank}: load weight end.")

    def profile_max_num_token(self, total_gpu_memory):
        available_gpu_memory = get_available_gpu_memory(
            self.tp_rank, distributed=self.tp_size > 1
        ) * (1 << 30)
        head_dim = (
            self.model_config.hidden_size // self.model_config.num_attention_heads
        )
        head_num = self.model_config.num_key_value_heads // self.tp_size
        cell_size = head_num * head_dim * self.model_config.num_hidden_layers * 2 * 2
        rest_memory = available_gpu_memory - total_gpu_memory * (
            1 - self.mem_fraction_static
        )
        max_num_token = int(rest_memory // cell_size)
        return max_num_token

    def init_memory_pool(self, total_gpu_memory):
        self.max_total_num_token = self.profile_max_num_token(total_gpu_memory)

        if self.max_total_num_token <= 0:
            raise RuntimeError(
                "Not enought memory. " "Please try to increase --mem-fraction-static."
            )

        self.req_to_token_pool = ReqToTokenPool(
            int(self.max_total_num_token / self.model_config.context_len * 256),
            self.model_config.context_len + 8,
        )
        self.token_to_kv_pool = TokenToKVPool(
            self.max_total_num_token,
            dtype=torch.float16,
            head_num=self.model_config.num_key_value_heads // self.tp_size,
            head_dim=self.model_config.hidden_size
            // self.model_config.num_attention_heads,
            layer_num=self.model_config.num_hidden_layers,
        )

    @torch.inference_mode()
    def forward_prefill(
        self,
        input_ids,
        req_pool_indices,
        seq_lens,
        position_ids_offsets,
        out_cache_loc,
        return_logprob,
    ):
        input_metadata = InputMetadata.create(
            self,
            forward_mode=ForwardMode.PREFILL,
            tp_size=self.tp_size,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            position_ids_offsets=position_ids_offsets,
            out_cache_loc=out_cache_loc,
            return_logprob=return_logprob,
        )
        return self.model.forward(input_ids, input_metadata.positions, input_metadata)

    @torch.inference_mode()
    def forward_decode(
        self,
        input_ids,
        req_pool_indices,
        seq_lens,
        position_ids_offsets,
        out_cache_loc,
        out_cache_cont_start,
        out_cache_cont_end,
    ):
        input_metadata = InputMetadata.create(
            self,
            forward_mode=ForwardMode.DECODE,
            tp_size=self.tp_size,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            position_ids_offsets=position_ids_offsets,
            out_cache_loc=out_cache_loc,
            out_cache_cont_start=out_cache_cont_start,
            out_cache_cont_end=out_cache_cont_end,
        )
        return self.model.forward(input_ids, input_metadata.positions, input_metadata)[
            0
        ]

    def forward(self, batch: Batch, forward_mode: ForwardMode, return_logprob=False):
        
        kwargs = {
            "input_ids": batch.input_ids,
            "req_pool_indices": batch.req_pool_indices,
            "seq_lens": batch.seq_lens,
            "position_ids_offsets": batch.position_ids_offsets,
            "out_cache_loc": batch.out_cache_loc,
        }

        if forward_mode == ForwardMode.DECODE:
            kwargs["out_cache_cont_start"] = batch.out_cache_cont_start
            kwargs["out_cache_cont_end"] = batch.out_cache_cont_end
            return self.forward_decode(**kwargs)
        elif forward_mode == ForwardMode.PREFILL:
            kwargs["return_logprob"] = return_logprob
            return self.forward_prefill(**kwargs)
        else:
            raise ValueError(f"Invaid forward mode: {forward_mode}")
