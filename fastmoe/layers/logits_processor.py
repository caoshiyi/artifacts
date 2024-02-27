import torch
from fastmoe.serve.router.model_runner import ForwardMode, InputMetadata
from torch import nn
from vllm.model_executor.parallel_utils.communication_op import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)


class LogitsProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()

    def forward(self, input_ids, hidden_states, weight, input_metadata):
        if not input_metadata.return_logprob:
            if input_metadata.forward_mode == ForwardMode.DECODE or input_metadata.forward_mode == ForwardMode.PARTIAL_DECODE:
                last_hidden = hidden_states
            else:
                last_index = (
                    torch.cumsum(
                        input_metadata.seq_lens,
                        dim=0,
                        dtype=torch.long,
                    )
                    - 1
                )
                last_hidden = hidden_states[last_index]
                hidden_states = None

            last_logits = torch.matmul(last_hidden, weight.T)
            if self.tp_size > 1:
                last_logits = tensor_model_parallel_all_gather(last_logits)
            last_logits = last_logits[:, : self.config.vocab_size]
            return last_logits, (None, None)
        else:
            assert input_metadata.forward_mode != ForwardMode.DECODE
            last_index = (
                torch.cumsum(
                    input_metadata.seq_lens,
                    dim=0,
                    dtype=torch.long,
                )
                - 1
            )

            logits = torch.matmul(hidden_states, weight.T)
            if self.tp_size > 1:
                logits = tensor_model_parallel_all_gather(logits)
            logits = logits[:, : self.config.vocab_size]
            all_logprobs = torch.log(torch.softmax(logits.float(), dim=-1) + 1e-6)

            logprobs = all_logprobs[
                torch.arange(all_logprobs.shape[0], device="cuda"),
                torch.cat([input_ids[1:], torch.tensor([0], device="cuda")]),
            ]
            logprobs_cumsum = torch.cumsum(logprobs, dim=0, dtype=torch.float32)

            start = input_metadata.extend_start_loc.clone()
            end = start + input_metadata.extend_seq_lens - 2
            start.clamp_(min=0, max=logprobs.shape[0] - 1)
            end.clamp_(min=0, max=logprobs.shape[0] - 1)
            sum_logp = logprobs_cumsum[end] - logprobs_cumsum[start] + logprobs[start]
            normalized_logprobs = sum_logp / (
                (input_metadata.extend_seq_lens - 1).clamp(min=1)
            )

            last_logits = logits[last_index]
            return last_logits, (logprobs, normalized_logprobs)