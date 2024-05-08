# coding=utf-8
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from tqdm import tqdm
import time

from fastmoe.serve.router.infer_batch import DecodePart
from fastmoe.layers.attention import Attention
from fastmoe.layers.stacked_fused_moe import stack_fused_moe
from fastmoe.layers.linear import (LinearMethodBase,
                                               QKVParallelLinear,
                                               StackedLinear,
                                               RowParallelLinear)
from fastmoe.layers.logits_processor import LogitsProcessor
from fastmoe.serve.router.model_runner import InputMetadata
from fastmoe.backend.memory import MemoryPool
from fastmoe.models.configs.dbrx import DbrxConfig


from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)



class DbrxRouter(nn.Module):
    """A Router implementation for DBRX that returns logits for each expert
    per token.
    """

    def __init__(
        self,
        config: DbrxConfig,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.ffn_config.moe_num_experts
        self.d_model = config.d_model
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
            print(f"Using default dtype {params_dtype}")
        self.params_dtype = params_dtype

        self.layers = StackedLinear(
            config.n_layers,
            self.d_model,
            self.num_total_experts,
            bias=False,
            params_dtype=self.params_dtype,
            linear_method=None
        )

    def forward(self, index:int, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.layers(index, hidden_states)
        return router_logits


class DbrxExperts(nn.Module):
    """A tensor-parallel MoE implementation for DBRX.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        config: DbrxConfig,
        params_dtype: Optional[torch.dtype] = None,
        memory_pool: Optional[MemoryPool] = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_layers = config.n_layers
        self.num_total_experts = config.ffn_config.moe_num_experts
        self.top_k = config.ffn_config.moe_top_k
        self.hidden_size = config.d_model
        self.intermediate_size = (config.ffn_config.ffn_hidden_size //
                                  self.tp_size)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.router = DbrxRouter(config, self.params_dtype)
        
        self.ws = nn.Parameter(
            torch.empty(self.num_layers, self.num_total_experts,
                        3 * self.intermediate_size * self.hidden_size,
                        device="cpu",
                        dtype=self.params_dtype))

        set_weight_attrs(
            self.ws,
            {
                "weight_loader": self.weight_loader,
            },
        )

        # memory manager
        set_weight_attrs(self.ws, {"mem": memory_pool})
        # mapping from cpu mem to gpu mem (expert id -> indices)
        # must use int64, since page size is large
        indices = torch.empty((self.num_layers, self.num_total_experts), dtype=torch.int64, device="cuda")
        indices.fill_(memory_pool.size)
        set_weight_attrs(self.ws, {"mapping": indices}) 

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, layer_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        v1_offset = shard_size * self.hidden_size
        w2_offset = 2 * shard_size * self.hidden_size
        # DBRX uses GLU for each experts.
        # GLU has 3 linear layers: w1, v1 and w2.
        if weight_name.endswith("w1"):
            loaded_weight = torch.reshape(
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.hidden_size],
            )
            param_data[layer_id, :, :v1_offset] = loaded_weight[:, shard, :].reshape(self.num_total_experts, -1)
        if weight_name.endswith("v1"):
            loaded_weight = torch.reshape(
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.hidden_size],
            )
            param_data[layer_id, :, v1_offset:w2_offset] = loaded_weight[:, shard, :].reshape(self.num_total_experts, -1)
        if weight_name.endswith("w2"):
            loaded_weight = torch.reshape(
                loaded_weight,
                [-1, self.intermediate_size * self.tp_size, self.hidden_size],
            ).transpose(1, 2)
            param_data[layer_id, :, w2_offset:] = loaded_weight[:, :, shard].reshape(self.num_total_experts, -1)

    def forward(self, index:int, hidden_states: torch.Tensor) -> torch.Tensor:
        # num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.router(index, hidden_states)
        final_hidden_states = stack_fused_moe(hidden_states,
                                                self.ws.mem.mem_data,
                                                self.ws.mapping[index],
                                                router_logits,
                                                self.top_k,
                                                renormalize=True,
                                                inplace=True)

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states


class DbrxAttention(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        layer_id: int = 0,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.total_num_heads = config.n_heads
        self.head_dim = self.d_model // self.total_num_heads
        self.total_num_kv_heads = config.attn_config.kv_n_heads
        self.clip_qkv = config.attn_config.clip_qkv
        self.rope_theta = config.attn_config.rope_theta
        self.max_position = config.max_seq_len

        # pylint: disable=invalid-name
        self.Wqkv = QKVParallelLinear(
            self.d_model,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.out_proj = RowParallelLinear(
            self.d_model,
            self.d_model,
            bias=False,
            linear_method=linear_method,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_world_size
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size
        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        if input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.PREATTN:
            qkv, _ = self.Wqkv(hidden_states)
            if self.clip_qkv is not None:
                qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.rotary_emb(position_ids, q, k)
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
            output, _ = self.out_proj(attn_output)
        return output


class DbrxFusedNormAttention(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        layer_id : int = 0,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.attn = DbrxAttention(config, layer_id, linear_method)
        self.norm_1 = nn.LayerNorm(self.d_model)
        self.norm_2 = nn.LayerNorm(self.d_model)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.PREATTN:
            residual = hidden_states
            hidden_states = self.norm_1(hidden_states)
        
        # Self Attention
        if input_metadata.decode_part == DecodePart.PREATTN:
            qkv = self.attn(position_ids, hidden_states, input_metadata)
            return qkv, residual
        elif input_metadata.decode_part == DecodePart.CPU_ATTN:
            attn_out = self.attn(position_ids, hidden_states, input_metadata)
            return attn_out
        elif input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.POSTATTN:
            x = self.attn(
                position_ids=position_ids,
                hidden_states=hidden_states,
                input_metadata=input_metadata,
            )
            hidden_states = residual + x
            residual = hidden_states
            hidden_states = self.norm_2(hidden_states)

        return hidden_states, residual


class DbrxBlock(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        layer_id: int = 0,
        dbrx_experts: DbrxExperts = None,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.norm_attn_norm = DbrxFusedNormAttention(config, layer_id, linear_method)
        self.ffn = dbrx_experts

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.POSTATTN:
            hidden_states, residual = self.norm_attn_norm(
                position_ids=position_ids,
                hidden_states=hidden_states,
                input_metadata=input_metadata,
                residual=residual,
            )
            hidden_states = self.ffn(self.layer_id, hidden_states)
            hidden_states = hidden_states + residual
            return hidden_states
        elif input_metadata.decode_part == DecodePart.PREATTN:
            qkv, residual = self.norm_attn_norm(
                position_ids=position_ids,
                hidden_states=hidden_states,
                input_metadata=input_metadata,
                residual=None,
            )
            return qkv, residual
        elif input_metadata.decode_part == DecodePart.CPU_ATTN:
            attn_out = self.norm_attn_norm(
                position_ids=position_ids,
                hidden_states=hidden_states,
                input_metadata=input_metadata,
                residual=None,
            )
            return attn_out


class DbrxModel(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        linear_method: Optional[LinearMethodBase] = None,
        memory_pool: Optional[MemoryPool] = None,
    ):
        super().__init__()
        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.d_model,
        )
        self.memory_pool = memory_pool
        self.dbrx_experts = DbrxExperts(config, memory_pool=memory_pool)
        self.blocks = nn.ModuleList(
            [DbrxBlock(config, i, dbrx_experts=self.dbrx_experts, linear_method=linear_method) for i in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model, eps=1e-5)
        for module in self.modules():
            if hasattr(module, "bias") and isinstance(module.bias,
                                                      nn.Parameter):
                # Remove the bias term in Linear and LayerNorm.
                module.register_parameter("bias", None)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        input_metadata: InputMetadata,
        hidden_states: torch.Tensor = None,
        residual: torch.Tensor = None,
        cur_layers: List[int] = None,
    ) -> torch.Tensor:
        if hidden_states is None and input_ids is not None:
            hidden_states = self.wte(input_ids)
        if cur_layers is None:
            cur_layers = range(len(self.blocks))
        for i in cur_layers:
            block = self.blocks[i]
            if input_metadata.decode_part == DecodePart.ALL or input_metadata.decode_part == DecodePart.POSTATTN:
                hidden_states = block(position_ids, hidden_states,
                                                input_metadata,
                                                residual)
            elif input_metadata.decode_part == DecodePart.PREATTN:
                qkv, residual = block(position_ids, hidden_states,
                                                input_metadata, None)
                return qkv, residual
            elif input_metadata.decode_part == DecodePart.CPU_ATTN:
                attn_out = block(None, None, input_metadata, None)
                return attn_out

        if cur_layers[-1] != len(self.blocks) - 1:
            return hidden_states, None
        else:
            hidden_states = self.norm_f(hidden_states)
            return hidden_states, None


class DbrxForCausalLMOff(nn.Module):

    def __init__(
        self,
        config: DbrxConfig,
        linear_method: Optional[LinearMethodBase] = None,
        memory_pool: Optional[MemoryPool] = None,
    ):
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.expert_pool = memory_pool
        self.unpadded_vocab_size = config.vocab_size
        self.transformer = DbrxModel(config, linear_method, memory_pool)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.d_model,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
        )
        self.logits_processor = LogitsProcessor(config)

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
            hidden_states, residual = self.transformer(input_ids, positions,
                                    input_metadata, hidden_states=hidden_states, residual=residual, cur_layers=cur_layers)
        elif input_metadata.decode_part == DecodePart.PREATTN:
            qkv, residual = self.transformer(input_ids, positions,
                                    input_metadata, hidden_states=hidden_states, residual=residual, cur_layers=cur_layers)
            return qkv, residual
        elif input_metadata.decode_part == DecodePart.CPU_ATTN:
            attn_out = self.transformer(input_ids, positions,
                                    input_metadata, hidden_states=hidden_states, residual=residual, cur_layers=cur_layers)
            return attn_out
        if self.config.n_layers - 1 not in cur_layers:
            return hidden_states, residual
        else:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head.weight, input_metadata
            )
    
    def prefetch_expert(self, experts: List[Tuple[int, int]]):
        start_pos = (experts[0][0] * self.config.ffn_config.moe_num_experts + experts[0][1]) % self.expert_pool.size
        end_pos = start_pos + len(experts)
        self.expert_pool.mem_data[start_pos:end_pos].copy_(self.transformer.dbrx_experts.ws.data[experts[0][0], experts[0][1]:experts[0][1]+len(experts)], non_blocking=True)
    
    def prefetch_experts_to_pin(self, layer_id, expert_id, partition_id, total_partition, wait_event):
        wait_event.synchronize()
        partition_size = self.expert_pool.mem_data.shape[1] // total_partition
        start = partition_id * partition_size
        end = min((partition_id + 1) * partition_size, self.expert_pool.mem_data.shape[1])
        self.experts_pin_buffer[expert_id, start:end].copy_(self.transformer.dbrx_experts.ws.data[layer_id, expert_id, start:end])
    
    def prefetch_experts_from_pin(self, layer_id, expert_id, partition_id, total_partition):
        buf_pos = (layer_id * self.config.ffn_config.moe_num_experts + expert_id) % self.expert_pool.size
        partition_size = self.expert_pool.mem_data.shape[1] // total_partition
        start = partition_id * partition_size
        end = min((partition_id + 1) * partition_size, self.expert_pool.mem_data.shape[1])
        self.expert_pool.mem_data[buf_pos, start:end].copy_(self.experts_pin_buffer[expert_id, start:end], non_blocking=True)
        
    def load_weights(self, 
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        expert_params_mapping = [
            # (param_name, weight_name, layer_id)
            ("dbrx_experts.ws",
             f"blocks.{layer_id}.ffn.experts.mlp.{weight_name}", layer_id)
            for weight_name in ["w1", "v1", "w2"]
            for layer_id in range(self.config.n_layers)
        ]
        gate_params_mapping = [
            # (param_name, weight_name, layer_id)
            ("dbrx_experts.router.layers.weight",
             f"blocks.{layer_id}.ffn.router.layer.weight", layer_id)
            for layer_id in range(self.config.n_layers)
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in tqdm(hf_model_weights_iterator(
                model_name_or_path,
                cache_dir,
                load_format,
                revision,
                fall_back_to_pt=False)):
            
            for (param_name, weight_name, layer_id) in gate_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, layer_id)
                break
            else:
                for param_name, weight_name, layer_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                    loaded_weight,
                                    weight_name,
                                    layer_id=layer_id)
                    break
                else:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
        
        for layer_id in range(self.config.n_layers):
            for expert_id in range(self.config.ffn_config.moe_num_experts):
                self.transformer.dbrx_experts.ws.mapping[layer_id, expert_id] = (layer_id * self.config.ffn_config.moe_num_experts + expert_id) % self.expert_pool.size
        
        self.experts_pin_buffer = torch.empty((self.config.ffn_config.moe_num_experts, 3 * self.config.ffn_config.ffn_hidden_size * self.config.d_model), dtype=torch.get_default_dtype(), device="cpu").pin_memory()

EntryClass = DbrxForCausalLMOff