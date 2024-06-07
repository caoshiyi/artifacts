import asyncio
import dataclasses
from typing import List

import transformers
import uvloop
import zmq
import zmq.asyncio
from fastmoe.utils.hf_transformers_utils import (
    get_config,
    get_context_length,
    get_processor,
    get_tokenizer,
)
from fastmoe.serve.io_struct import (
    BatchStrOut,
    GenerateReqInput,
    TokenizedGenerateReqInput,
    BatchTokenizedGenerateReqInput,
)
from fastmoe.serve.sampling_params import SamplingParams
from fastmoe.serve.server_args import PortArgs, ServerArgs

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@dataclasses.dataclass
class ReqState:
    out_list: List
    finished: bool
    event: asyncio.Event
    lock: asyncio.Lock


global global_processor


def init_global_processor(server_args: ServerArgs):
    global global_processor
    transformers.logging.set_verbosity_error()
    global_processor = get_processor(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )

class TokenizerManager:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = context.socket(zmq.PULL)
        self.recv_from_detokenizer.bind(f"tcp://127.0.0.1:{port_args.tokenizer_port}")

        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{port_args.router_port}")

        self.model_path = server_args.model_path
        self.hf_config = get_config(
            self.model_path, trust_remote_code=server_args.trust_remote_code
        )

        self.context_len = get_context_length(self.hf_config)

        self.tokenizer = get_tokenizer(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
        )

        self.to_create_loop = True
        self.rid_to_state = {}  # Dict[str -> ReqState]

    async def generate_request(self, obj: GenerateReqInput):
        if self.to_create_loop:
            await self.create_handle_loop()

        is_single = isinstance(obj.text, str)

        if is_single:
            rid = obj.rid
            input_ids = self.tokenizer.encode(obj.text)
            sampling_params = SamplingParams(**obj.sampling_params)
            if sampling_params.max_new_tokens != 0:
                sampling_params.normalize(self.tokenizer)
                sampling_params.verify()

            tokenized_obj = TokenizedGenerateReqInput(
                rid=rid,
                input_text=obj.text,
                input_ids=input_ids,
                sampling_params=sampling_params,
                return_logprob=obj.return_logprob,
                logprob_start_len=obj.logprob_start_len,
                stream=obj.stream,
            )
            self.send_to_router.send_pyobj(tokenized_obj)

            lock = asyncio.Lock()
            event = asyncio.Event()
            state = ReqState([], False, event, lock)
            self.rid_to_state[rid] = state

            while True:
                await event.wait()
                yield state.out_list[-1]
                state.out_list = []
                if state.finished:
                    del self.rid_to_state[rid]
                    break
                event.clear()
        else:
            assert obj.stream is False
            if obj.batch:
                # in this mode, we assume the batch share the same sampling_params
                rid = obj.rid
                input_ids = self.tokenizer(obj.text).input_ids
                sampling_params = SamplingParams(**obj.sampling_params[0])
                if sampling_params.max_new_tokens != 0:
                        sampling_params.normalize(self.tokenizer)
                        sampling_params.verify()
                batch_tokenized_obj = BatchTokenizedGenerateReqInput(
                    rid=rid,
                    input_text=obj.text,
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    return_logprob=obj.return_logprob[0],
                    logprob_start_len=obj.logprob_start_len[0],
                    stream=obj.stream,
                )
                self.send_to_router.send_pyobj(batch_tokenized_obj)
                bs = len(obj.text)
                for i in range(bs):
                    lock = asyncio.Lock()
                    event = asyncio.Event()
                    state = ReqState([], False, event, lock)
                    self.rid_to_state[obj.rid[i]] = state
            else:
                bs = len(obj.text)
                for i in range(bs):
                    rid = obj.rid[i]
                    input_ids = self.tokenizer.encode(obj.text[i])
                    sampling_params = SamplingParams(**obj.sampling_params[i])
                    if sampling_params.max_new_tokens != 0:
                        sampling_params.normalize(self.tokenizer)
                        sampling_params.verify()
                    tokenized_obj = TokenizedGenerateReqInput(
                        rid=rid,
                        input_text=obj.text[i],
                        input_ids=input_ids,
                        sampling_params=sampling_params,
                        return_logprob=obj.return_logprob[i],
                        logprob_start_len=obj.logprob_start_len[i],
                        stream=obj.stream,
                    )
                    self.send_to_router.send_pyobj(tokenized_obj)

                    lock = asyncio.Lock()
                    event = asyncio.Event()
                    state = ReqState([], False, event, lock)
                    self.rid_to_state[rid] = state

            output_list = []
            for i in range(bs):
                rid = obj.rid[i]
                state = self.rid_to_state[rid]
                await state.event.wait()
                output_list.append(state.out_list[-1])
                assert state.finished
                del self.rid_to_state[rid]

            yield output_list

    async def create_handle_loop(self):
        self.to_create_loop = False
        loop = asyncio.get_event_loop()
        loop.create_task(self.handle_loop())

    async def handle_loop(self):
        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()

            if isinstance(recv_obj, BatchStrOut):
                for i, rid in enumerate(recv_obj.rids):
                    recv_obj.meta_info[i]["id"] = rid
                    out_dict = {
                        "text": recv_obj.output_str[i],
                        "meta_info": recv_obj.meta_info[i],
                    }
                    state = self.rid_to_state[rid]
                    state.out_list.append(out_dict)
                    state.finished = recv_obj.finished[i]
                    state.event.set()
            else:
                raise ValueError(f"Invalid object: {recv_obj}")
