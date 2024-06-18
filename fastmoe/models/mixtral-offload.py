# coding=utf-8
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mixtral.py
"""Inference-only Mixtral model."""
from typing import List, Optional, Tuple
import torch
from torch import nn
from transformers import MixtralConfig
from tqdm import tqdm
import time

from fastmoe.backend.task_meta import InputMetadata, DecodePart
from fastmoe.layers.attention import Attention
from fastmoe.layers.stacked_fused_moe import stack_fused_moe
from fastmoe.layers.linear import (LinearMethodBase,
                                               QKVParallelLinear,
                                               StackedLinear,
                                               RowParallelLinear)
from fastmoe.layers.logits_processor import LogitsProcessor


from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)

class MixtralMoE(nn.Module):
    """A tensor-parallel MoE implementation for Mixtral that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        tp_size: Optional[int] = None,
    ):
        super().__init__()
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.num_layers = num_layers
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
            print(f"Using default dtype {params_dtype}")
        self.params_dtype = params_dtype

        self.gates = StackedLinear(self.num_layers,
                                    self.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     params_dtype=self.params_dtype,
                                     linear_method=None)
        self.ws = nn.Parameter(
            torch.empty(self.num_layers, self.num_total_experts,
                        3 * self.intermediate_size * self.hidden_size,
                        device="cpu",
                        dtype=self.params_dtype))

        set_weight_attrs(self.ws, {
            "weight_loader": self.weight_loader,
        })

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, expert_id: int, layer_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        w3_offset = shard_size * self.hidden_size
        w2_offset = 2 * shard_size * self.hidden_size
        if weight_name.endswith("w1.weight"):
            param_data[layer_id, expert_id, 0 : w3_offset] = loaded_weight[shard, :].view(-1)
        if weight_name.endswith("w3.weight"):
            param_data[layer_id, expert_id,
                       w3_offset : w2_offset] = loaded_weight[shard, :].view(-1)
        if weight_name.endswith("w2.weight"):
            param_data[layer_id, expert_id, w2_offset:] = loaded_weight[:, shard].reshape(-1)

    def forward(self, index:int, hidden_states: torch.Tensor, experts_cache: torch.Tensor) -> torch.Tensor:
        # router_logits: (n_token, n_experts)
        router_logits, _ = self.gates(index, hidden_states)
        final_hidden_states = stack_fused_moe(hidden_states,
                                        self.ws.gpu_cache,
                                        experts_cache,
                                        router_logits,
                                        self.top_k,
                                        renormalize=True,
                                        inplace=True)


        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states


class MixtralAttention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 layer_id: int = 0,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 linear_method: Optional[LinearMethodBase] = None,
                 sliding_window: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        if input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.PREATTN:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.rotary_emb(positions, q, k)
            if input_metadata.decode_part == DecodePart.PREATTN:
                return torch.cat([q, k, v], dim=-1).contiguous()
            
        if input_metadata.decode_part == DecodePart.ALL:
            attn_output = self.attn(q, k, v, input_metadata)
            input_metadata.attn_event.record(torch.cuda.current_stream())
        if input_metadata.decode_part == DecodePart.CPU_ATTN:
            attn_output = self.attn(None, None, None, input_metadata)
            return attn_output
            
        if input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.POSTATTN:
            if input_metadata.decode_part == DecodePart.POSTATTN:
                attn_output = hidden_states
            output, _ = self.o_proj(attn_output)
        return output


class MixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        layer_id: int = 0,
        block_sparse_moe: MixtralMoE = None,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = MixtralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            layer_id=layer_id,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            sliding_window=config.sliding_window,
            linear_method=linear_method)
        self.block_sparse_moe = block_sparse_moe
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.PREATTN:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)
        
        if input_metadata.decode_part == DecodePart.PREATTN:
            qkv = self.self_attn(positions, hidden_states, input_metadata)
            return qkv, residual
        elif input_metadata.decode_part == DecodePart.CPU_ATTN:
            attn_out = self.self_attn(positions, hidden_states, input_metadata)
            return attn_out
        elif input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.POSTATTN:
            # Self Attention
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                input_metadata=input_metadata,
            )
            # Fully Connected
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
            hidden_states = self.block_sparse_moe(self.layer_id, hidden_states, input_metadata.experts_mapping)

        return hidden_states, residual


class MixtralModel(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        # init only one MixtralMoE
        self.block_sparse_moe = MixtralMoE(
            num_layers = config.num_hidden_layers,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size)
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(config, i, block_sparse_moe=self.block_sparse_moe, linear_method=linear_method)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        hidden_states: torch.Tensor = None,
        residual: torch.Tensor = None,
        cur_layers: List[int] = None,
    ) -> torch.Tensor:
        if hidden_states is None and input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        if cur_layers is None:
            cur_layers = range(len(self.layers))
        for i in cur_layers:
            layer = self.layers[i]
            if input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.POSTATTN:
                hidden_states, residual = layer(positions, hidden_states,
                                                input_metadata,
                                                residual)
            elif input_metadata.decode_part == DecodePart.PREATTN:
                qkv, residual = layer(positions, hidden_states,
                                                input_metadata,
                                                residual)
                return qkv, residual
            elif input_metadata.decode_part == DecodePart.CPU_ATTN:
                attn_out = layer(None, None, input_metadata, None)
                return attn_out
        if cur_layers[-1] != len(self.layers) - 1:
            return hidden_states, residual
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states, _


class MixtralForCausalLMOff(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = MixtralModel(config, linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)
        self.tp_size = get_tensor_model_parallel_world_size()
        # self.current_experts = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
        hidden_states: torch.Tensor = None,
        residual: torch.Tensor = None,
        cur_layers: List[int] = None,
    ) -> torch.Tensor:
        if input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.POSTATTN:
            if hidden_states is not None:
                hidden_states = hidden_states.view(hidden_states.shape[0], -1)
            hidden_states, residual = self.model(input_ids, positions,
                                    input_metadata, hidden_states=hidden_states, residual=residual, cur_layers=cur_layers)
        elif input_metadata.decode_part == DecodePart.PREATTN:
            qkv, residual = self.model(input_ids, positions,
                                    input_metadata, hidden_states=hidden_states, residual=residual, cur_layers=cur_layers)
            return qkv, residual
        elif input_metadata.decode_part == DecodePart.CPU_ATTN:
            attn_out = self.model(input_ids, positions,
                                    input_metadata, hidden_states=hidden_states, residual=residual, cur_layers=cur_layers)
            return attn_out
        if self.config.num_hidden_layers - 1 not in cur_layers:
            return hidden_states, residual
        else:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head.weight, input_metadata
            )

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        expert_params_mapping = [
            # (param_name, weight_name, expert_id)
            ("block_sparse_moe.ws",
             f"layers.{layer_id}.block_sparse_moe.experts.{expert_id}.{weight_name}.weight", expert_id, layer_id)
            for expert_id in range(self.config.num_local_experts)
            for weight_name in ["w1", "w2", "w3"]
            for layer_id in range(self.config.num_hidden_layers)
        ]

        gate_params_mapping = [
            # (param_name, weight_name, layer_id)
            ("block_sparse_moe.gates.weight",
             f"layers.{layer_id}.block_sparse_moe.gate.weight", layer_id)
            for layer_id in range(self.config.num_hidden_layers)
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in tqdm(hf_model_weights_iterator(
                model_name_or_path,
                cache_dir,
                load_format,
                revision,
                fall_back_to_pt=False)):
            if "rotary_emb.inv_freq" in name:
                continue

            for (param_name, weight_name, layer_id) in gate_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, layer_id)
                break
            else:
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    for param_name, weight_name, expert_id, layer_id in expert_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(param,
                                    loaded_weight,
                                    weight_name,
                                    expert_id=expert_id,
                                    layer_id=layer_id)
                        break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
    
    def get_experts_mem(self):
        return self.model.block_sparse_moe.ws.data
    
    def link_gpu_experts_cache(self, expert_pool):
        set_weight_attrs(self.model.block_sparse_moe.ws, {"gpu_cache": expert_pool})
    

EntryClass = MixtralForCausalLMOff