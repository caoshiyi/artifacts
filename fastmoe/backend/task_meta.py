from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
import torch

from fastmoe.backend.memory import TokenToKVPool

class ForwardMode(Enum):
    PREFILL = auto()
    DECODE = auto()

class DecodePart(Enum):
    CPU_ATTN = auto()
    PREATTN = auto()
    POSTATTN = auto()
    ALL = auto()
    


@dataclass
class InputMetadata:
    forward_mode: ForwardMode
    decode_part: DecodePart = None

    seq_lens: torch.Tensor = None
    start_loc: torch.Tensor = None
    max_seq_len: int = None
    positions: torch.Tensor = None
    total_num_tokens: int = 0
    
    # mem and index
    token_to_kv_pool: TokenToKVPool = None
    cpu_token_start_index: int = None
    out_cache_loc: torch.Tensor = None
    # attn input
    qkv_pin: torch.Tensor = None
    # attn output
    hidden_pin: torch.Tensor = None
    # experts_mapping
    experts_mapping: torch.Tensor = None

    # sync
    attn_event: torch.cuda.Event = None

    # other
    other_kv_index: torch.Tensor = None
    return_logprob: bool = False
    
    @classmethod
    def create(
        cls,
        forward_mode,
        decode_part,
        seq_lens = None,
        positions = None,
        start_loc = None,
        max_seq_len = None,
        total_num_tokens = None,
        out_cache_loc = None,
        token_to_kv_pool = None,
        return_logprob=False,
        attn_event=None,
        experts_mapping=None,
    ):
        # batch_size = len(seq_lens)
        # if decode_part == DecodePart.ALL:
        #     start_loc = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
        #     start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)
        #     total_num_tokens = int(torch.sum(seq_lens))
        #     max_seq_len = int(torch.max(seq_lens))

        if forward_mode == ForwardMode.DECODE:
            if decode_part == DecodePart.PREATTN:
                positions = ((seq_lens - 1)).to(torch.int64)
        else:
            other_kv_index = None

        if decode_part == DecodePart.ALL:
            ret = cls(
                forward_mode=forward_mode,
                decode_part=decode_part,
                max_seq_len=max_seq_len,
                start_loc=start_loc,
                seq_lens=seq_lens,
                positions=positions,
                total_num_tokens=total_num_tokens,
                out_cache_loc=out_cache_loc,
                token_to_kv_pool=token_to_kv_pool,
                return_logprob=return_logprob,
                other_kv_index=other_kv_index,
                attn_event=attn_event,
                experts_mapping=experts_mapping,
            )
        elif decode_part == DecodePart.PREATTN:
            ret = cls(
                forward_mode=forward_mode,
                decode_part=decode_part,
                positions=positions,
            )
        elif decode_part == DecodePart.POSTATTN:
            ret = cls(
                forward_mode=forward_mode,
                decode_part=decode_part,
                experts_mapping=experts_mapping,
                return_logprob=return_logprob,
            )

        return ret
    
    @classmethod
    def create_cpu_attn(
        cls,
        forward_mode,
        decode_part,
        cpu_token_start_index,
        start_loc,
        seq_lens,
        out_cache_loc,
        qkv_pin,
        hidden_pin,
        token_to_kv_pool,
    ):

        ret = cls(
            forward_mode=forward_mode,
            decode_part=decode_part,
            start_loc=start_loc,
            seq_lens=seq_lens,
            token_to_kv_pool=token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            cpu_token_start_index=cpu_token_start_index,
            qkv_pin=qkv_pin,
            hidden_pin=hidden_pin,
        )

        return ret