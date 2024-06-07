import asyncio
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from typing import List, Optional, Union

# Fix a Python bug
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import aiohttp
import psutil
import requests
import uvicorn
import uvloop
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from fastmoe.serve.conversation import (
    Conversation,
    SeparatorStyle,
    chat_template_exists,
    generate_chat_conv,
    register_conv_template,
)
from fastmoe.utils.hf_transformers_utils import get_tokenizer
from fastmoe.serve.detokenizer_manager import start_detokenizer_process
from fastmoe.serve.io_struct import GenerateReqInput
from fastmoe.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    UsageInfo,
)
from fastmoe.serve.router.manager import start_router_process
from fastmoe.serve.tokenizer_manager import TokenizerManager
from fastmoe.serve.server_args import PortArgs, ServerArgs
from fastmoe.utils.utils import handle_port_init

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


app = FastAPI()
tokenizer_manager = None
chat_template_name = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

async def stream_generator(obj):
    async for out in tokenizer_manager.generate_request(obj):
        yield out


@app.post("/generate")
async def generate_request(obj: GenerateReqInput):
    obj.post_init()

    if obj.stream:

        async def stream_results():
            async for out in stream_generator(obj):
                yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_results(), media_type="text/event-stream")

    ret = await tokenizer_manager.generate_request(obj).__anext__()
    return ret

def launch_server(server_args, pipe_finish_writer):
    global tokenizer_manager
    global chat_template_name

    # Handle ports
    server_args.port, server_args.additional_ports = handle_port_init(
        server_args.port, server_args.additional_ports, server_args.tp_size
    )

    port_args = PortArgs(
        tokenizer_port=server_args.additional_ports[0],
        router_port=server_args.additional_ports[1],
        detokenizer_port=server_args.additional_ports[2],
        nccl_port=server_args.additional_ports[3],
        model_rpc_ports=server_args.additional_ports[4:],
    )

    # Load chat template if needed
    if server_args.chat_template is not None:
        print(f"Use chat template: {server_args.chat_template}")
        if not chat_template_exists(server_args.chat_template):
            if not os.path.exists(server_args.chat_template):
                raise RuntimeError(
                    f"Chat template {server_args.chat_template} is not a built-in template name "
                    "or a valid chat template file path."
                )
            with open(server_args.chat_template, "r") as filep:
                template = json.load(filep)
                try:
                    sep_style = SeparatorStyle[template["sep_style"]]
                except KeyError:
                    raise ValueError(
                        f"Unknown separator style: {template['sep_style']}"
                    ) from None
                register_conv_template(
                    Conversation(
                        name=template["name"],
                        system_template=template["system"] + "\n{system_message}",
                        system_message=template.get("system_message", ""),
                        roles=(template["user"], template["assistant"]),
                        sep_style=sep_style,
                        sep=template.get("sep", "\n"),
                        stop_str=template["stop_str"],
                    ),
                    override=True,
                )
            chat_template_name = template["name"]
        else:
            chat_template_name = server_args.chat_template

    # Launch processes
    tokenizer_manager = TokenizerManager(server_args, port_args)
    pipe_router_reader, pipe_router_writer = mp.Pipe(duplex=False)
    pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)

    proc_router = mp.Process(
        target=start_router_process,
        args=(
            server_args,
            port_args,
            pipe_router_writer,
        ),
    )
    proc_router.start()
    proc_detoken = mp.Process(
        target=start_detokenizer_process,
        args=(
            server_args,
            port_args,
            pipe_detoken_writer,
        ),
    )
    proc_detoken.start()

    # Wait for the model to finish loading
    router_init_state = pipe_router_reader.recv()
    detoken_init_state = pipe_detoken_reader.recv()

    if router_init_state != "init ok" or detoken_init_state != "init ok":
        proc_router.kill()
        proc_detoken.kill()
        print("router init state:", router_init_state)
        print("detoken init state:", detoken_init_state)
        sys.exit(1)

    assert proc_router.is_alive() and proc_detoken.is_alive()

    def launch_server():
        # Launch api server
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )

    t = threading.Thread(target=launch_server)
    t.start()

    if pipe_finish_writer:
        url = server_args.url()

        success = False
        for i in range(60):
            time.sleep(1)
            try:
                res = requests.get(url + "/get_model_info", timeout=5)
                success = True
                break
            except requests.exceptions.RequestException as e:
                pass

        if success:
            pipe_finish_writer.send("init ok")
        else:
            pipe_finish_writer.send(str(e))


class Runtime:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        load_format: str = "auto",
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = True,
        cpu_mem_bdw: float = None,
        avg_prompt_len: int = 32,
        gen_len: int = 1,
        mem_fraction_static: float = ServerArgs.mem_fraction_static,
        max_prefill_num_token: int = ServerArgs.max_prefill_num_token,
        tp_size: int = 1,
        random_seed: int = 42,
        log_level: str = "error",
        port: Optional[int] = None,
        additional_ports: Optional[Union[List[int], int]] = None,
    ):
        host = "127.0.0.1"
        port, additional_ports = handle_port_init(port, additional_ports, tp_size)
        self.server_args = ServerArgs(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            host=host,
            port=port,
            additional_ports=additional_ports,
            load_format=load_format,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            cpu_mem_bdw=cpu_mem_bdw,
            avg_prompt_len=avg_prompt_len,
            gen_len=gen_len,
            mem_fraction_static=mem_fraction_static,
            max_prefill_num_token=max_prefill_num_token,
            tp_size=tp_size,
            random_seed=random_seed,
            log_level=log_level,
        )

        self.url = self.server_args.url()
        self.generate_url = (
            f"http://{self.server_args.host}:{self.server_args.port}/generate"
        )

        self.pid = None
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)
        proc = mp.Process(target=launch_server, args=(self.server_args, pipe_writer))
        proc.start()
        pipe_writer.close()
        self.pid = proc.pid

        try:
            init_state = pipe_reader.recv()
        except EOFError:
            init_state = ""

        if init_state != "init ok":
            self.shutdown()
            raise RuntimeError("Launch failed. Please see the error messages above.")

        self.endpoint = RuntimeEndpoint(self.url)

    def shutdown(self):
        if self.pid is not None:
            try:
                parent = psutil.Process(self.pid)
            except psutil.NoSuchProcess:
                return
            children = parent.children(recursive=True)
            for child in children:
                child.kill()
            psutil.wait_procs(children, timeout=5)
            parent.kill()
            parent.wait(timeout=5)
            self.pid = None

    def get_tokenizer(self):
        return get_tokenizer(
            self.server_args.tokenizer_path,
            tokenizer_mode=self.server_args.tokenizer_mode,
            trust_remote_code=self.server_args.trust_remote_code,
        )

    async def add_request(
        self,
        prompt: str,
        sampling_params,
    ) -> None:
        json_data = {
            "text": prompt,
            "sampling_params": sampling_params,
            "stream": True,
        }

        pos = 0

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(self.generate_url, json=json_data) as response:
                async for chunk, _ in response.content.iter_chunks():
                    chunk = chunk.decode("utf-8")
                    if chunk and chunk.startswith("data:"):
                        if chunk == "data: [DONE]\n\n":
                            break
                        data = json.loads(chunk[5:].strip("\n"))
                        cur = data["text"][pos:]
                        if cur:
                            yield cur
                        pos += len(cur)

    def __del__(self):
        self.shutdown()
