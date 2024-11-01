import os
from typing import Optional, Union

import torch
from hf_transformers_utils import (get_config, get_context_length, 
                                                get_num_layers, get_hidden_size, 
                                                get_num_attention_heads, get_num_kv_heads, 
                                                get_intermediate_size, get_num_experts, get_topk)


class ModelConfig:
    def __init__(
        self,
        path: str,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
    ) -> None:
        self.path = path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.hf_config = get_config(self.path, trust_remote_code, revision)

        # Unify the config keys for hf_config
        self.context_len = get_context_length(self.hf_config)
        self.head_dim =  get_hidden_size(self.hf_config) // get_num_attention_heads(self.hf_config)
        self.num_attention_heads = get_num_attention_heads(self.hf_config)
        self.num_key_value_heads = get_num_kv_heads(self.hf_config)
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.hidden_size = get_hidden_size(self.hf_config)
        self.num_hidden_layers = get_num_layers(self.hf_config)
        self.vocab_size = self.hf_config.vocab_size
        self.num_local_experts = get_num_experts(self.hf_config)
        self.intermediate_size = get_intermediate_size(self.hf_config)
        self.topk = get_topk(self.hf_config)
