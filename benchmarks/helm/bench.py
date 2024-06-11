import argparse
import numpy as np
import json
import requests
import time
from helm.benchmark.presentation.run_entry import RunEntry
from helm.benchmark.run import run_entries_to_run_specs
from helm.benchmark.run_specs import (ScenarioSpec, RunSpec, get_summarization_adapter_spec,
    get_summarization_metric_specs, get_generative_harms_metric_specs,
    ADAPT_MULTIPLE_CHOICE_JOINT, get_multiple_choice_adapter_spec)
from helm.benchmark.runner import (create_scenario, AdapterFactory, with_instance_ids, create_metric,
    TokensMetric, Metric, MetricSpec, MetricResult, PerInstanceStats, create_metric, Stat,
    ScenarioState, Counter, MetricName, ensure_directory_exists, write, asdict_without_nones,
    DataPreprocessor)
from helm.common.request import Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (TokenizationRequestResult,
    TokenizationRequest, TokenizationToken, DecodeRequest, DecodeRequestResult)
from helm.proxy.clients.client import truncate_sequence

from transformers import AutoTokenizer, AutoConfig
# pip install lightning==2.0.1 crfm-helm==0.2.1 transformers==4.41.2

class MixtralTokenizer:
    # Adapted from helm/proxy/clients/huggingface_client.py

    def __init__(self, name):
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.tokenizer.add_bos_token = False

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        tokenizer = self.tokenizer

        def do_it():
            if request.encode:
                if request.truncation:
                    tokens = tokenizer.encode(
                        request.text,
                        truncation=request.truncation,
                        max_length=request.max_length,
                        add_special_tokens=False,
                    )
                else:
                    tokens = tokenizer.encode(request.text, add_special_tokens=False)
            else:
                tokens = tokenizer.tokenize(request.text)
            return {"tokens": tokens}

        result = do_it()

        return TokenizationRequestResult(
            success=True,
            cached=False,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=0,
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        tokenizer = self.tokenizer


        def do_it():
            return {
                "text": tokenizer.decode(
                    request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces
                )
            }

        result = do_it()

        return DecodeRequestResult(
            success=True, cached=False, text=result["text"], request_time=0,
        )

def get_info(scenario_state, tokenizer=None):
    prompts = []
    for r in scenario_state.request_states:
        prompts.append(r.request.prompt)

    if tokenizer is not None:
        # Tokenize
        input_ids = tokenizer(prompts).input_ids
        # prompts len sum
        avg_len = sum(np.array([len(input_id) for input_id in input_ids])) / len(input_ids)
        print(f"Average sequence length: {avg_len}")
        max_len = max([len(input_id) for input_id in input_ids])
        print(f"Max sequence length: {max_len}")

    return prompts
    

   


def execute(scenario_state, tokenizer):
    return get_info(scenario_state, tokenizer)


def get_requests(description, args):
    ##### RunSpec #####
    run_entries = [RunEntry(description, priority=1, groups=None)]
    run_specs = run_entries_to_run_specs(
        run_entries=run_entries,
        max_eval_instances=args.max_eval_instances,
        num_train_trials=1,
    )
    run_spec = run_specs[0]

    tokenizer_service = MixtralTokenizer(args.model)
    tokenizer = tokenizer_service.tokenizer
    adapter = AdapterFactory.get_adapter(run_spec.adapter_spec, tokenizer_service)

    ##### Scenario #####
    scenario = create_scenario(run_spec.scenario_spec)
    instances = scenario.get_instances()

    # Give each instance a unique ID
    instances = with_instance_ids(instances)

    # Data preprocessing
    instances = DataPreprocessor(run_spec.data_augmenter_spec).preprocess(
        instances, parallelism=1
    )
    scenario_state = adapter.adapt(instances, parallelism=1)
    
    return execute(scenario_state, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, required=True)
    parser.add_argument("--pad-len", type=int, default=256)
    parser.add_argument("--max-eval-instances", type=int, default=10)
    parser.add_argument("--num-req", type=int, default=100)
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()
    url = f"{args.host}:{args.port}"

    prompts = get_requests(args.description, args)

    warmup = requests.post(
        url + "/generate",
        json={
            "text": prompts[:20],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 2,
            },
            "batch": True,
            "max_padding_length": 0,
        },
    )
    print(warmup.json())

    max_new_tokens = 50
    max_padding_length = args.pad_len
    start = time.time()
    response = requests.post(
        url + "/generate",
        json={
            "text": prompts[:args.num_req],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
            "batch": True,
            "max_padding_length": max_padding_length,
        },
    )
    end = time.time()
    # save the result, create the file if it doesn't exist
    print(response.json())
    num_finished = 0
    for req in response.json():
         if req['meta_info']['completion_tokens'] == max_new_tokens:
            num_finished += 1
        
    print(f"Time: {end - start:.3f}s")
    print(f"Throughput: {max_new_tokens*num_finished / (end - start):.3f} tokens/s")
    results = {"time": end - start, "throughput": max_new_tokens*num_finished / (end - start), "max_new_tokens": max_new_tokens, "max_padding_length": max_padding_length}
    # write the results to a file, create the file if it doesn't exist, write to the end if it does
    with open("mtbench_moel_L4.jsonl", "a") as fout:
        fout.write(json.dumps(results) + "\n")