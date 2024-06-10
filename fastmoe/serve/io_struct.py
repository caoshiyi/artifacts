import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from fastmoe.serve.sampling_params import SamplingParams


@dataclass
class GenerateReqInput:
    # The input prompt
    text: Union[List[str], str]
    # The sampling_params
    sampling_params: Union[List[Dict], Dict] = None
    # The request id
    rid: Optional[Union[List[str], str]] = None
    # Whether return logprobs of the prompts
    return_logprob: Optional[Union[List[bool], bool]] = None
    # The start location of the prompt for return_logprob
    logprob_start_len: Optional[Union[List[int], int]] = None
    # Whether to stream output
    stream: bool = False
    batch: bool = False
    max_padding_length: int = 0

    def post_init(self):
        is_single = isinstance(self.text, str)

        if is_single:
            if self.sampling_params is None:
                self.sampling_params = {}
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.return_logprob is None:
                self.return_logprob = False
            if self.logprob_start_len is None:
                self.logprob_start_len = 0
        else:
            num = len(self.text)

            if self.sampling_params is None:
                self.sampling_params = [{}] * num
            elif not isinstance(self.sampling_params, list):
                self.sampling_params = [self.sampling_params] * num

            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(num)]
            else:
                assert isinstance(self.rid, list)

            if self.return_logprob is None:
                self.return_logprob = [False] * num
            elif not isinstance(self.return_logprob, list):
                self.return_logprob = [self.return_logprob] * num

            if self.logprob_start_len is None:
                self.logprob_start_len = [0] * num
            elif not isinstance(self.logprob_start_len, list):
                self.logprob_start_len = [self.logprob_start_len] * num


@dataclass
class TokenizedGenerateReqInput:
    rid: str
    input_text: str
    input_ids: List[int]
    sampling_params: SamplingParams
    return_logprob: bool
    logprob_start_len: int
    stream: bool

@dataclass
class BatchTokenizedGenerateReqInput:
    rid: List[str]
    input_text: List[str]
    input_ids: List[List[int]]
    sampling_params: SamplingParams
    return_logprob: bool
    logprob_start_len: int
    stream: bool

@dataclass
class BatchTokenIDOut:
    rids: List[str]
    output_tokens: List[List[int]]
    hit_stop_str: List[Optional[str]]
    skip_special_tokens: List[bool]
    meta_info: List[Dict]
    finished: List[bool]


@dataclass
class BatchStrOut:
    rids: List[str]
    output_str: List[str]
    meta_info: List[Dict]
    finished: List[bool]
