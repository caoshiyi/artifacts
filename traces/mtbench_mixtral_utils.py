import functools
import itertools
import os
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def parse_input_ids(data):
    matches = re.findall(r"input_ids: tensor\(\[\[([\s\d,]+)", data)
    return [[int(s.strip()) for s in match.split(",")] for match in matches]


def parse_expert_ids(data):
    matches = re.findall(
        r"Layer (\d+) topk_ids: tensor\(\[([\[\]\d,\s]+)\], device", data
    )
    layer_ids = [int(t[0]) for t in matches]
    expert_ids = [
        [[int(x) for x in s.strip()[1:-1].split(", ")] for s in t[1].split(",\n")]
        for t in matches
    ]
    return layer_ids, expert_ids


def parse_file(filename):
    with open(filename, "r") as f:
        data = f.read()

    input_ids = parse_input_ids(data)
    layer_ids, expert_ids = parse_expert_ids(data)
    assert len(layer_ids) == len(
        expert_ids
    ), f"len(layer_ids)={len(layer_ids)} and len(expert_ids)={len(expert_ids)}"
    assert len(layer_ids) == len(
        input_ids
    ), f"len(layer_ids)={len(layer_ids)} and len(input_ids)={len(input_ids)}"

    return layer_ids, input_ids, expert_ids


def load_all(path: Optional[str] = None) -> dict[str, "QueryTrace"]:
    if path is None:
        lib_path = os.path.dirname(__file__)
        path = os.path.join(lib_path, "MTBench_Mixtral")

    filenames = os.listdir(path)
    traces = {}
    for filename in filenames:
        filename = os.path.join(path, filename)
        try:
            trace = QueryTrace.from_file(filename)
        except AssertionError:
            pass
        traces[filename] = trace

    return traces


@dataclass(frozen=True)
class QueryTrace:
    """Trace for a query to the LLM."""

    layer_ids: Sequence[int]
    expert_ids: Sequence[Sequence[Sequence[int]]]  # layer, token, selected experts
    input_ids: Sequence[Sequence[int]]  # layer, inputs
    has_prefix: bool = True

    @staticmethod
    def from_file(filename):
        layer_ids, input_ids, expert_ids = parse_file(filename)
        return QueryTrace(
            layer_ids=layer_ids, expert_ids=expert_ids, input_ids=input_ids
        )

    @functools.cached_property
    def num_layers(self) -> int:
        return len(np.unique(self.layer_ids))

    @functools.cached_property
    def num_experts(self) -> int:
        exp_ids_flat = list(
            itertools.chain.from_iterable(
                itertools.chain.from_iterable(self.expert_ids)
            )
        )
        return len(np.unique(exp_ids_flat))

    @functools.cached_property
    def num_tokens(self) -> int:
        return sum(len(x) for x in self.input_ids)

    @functools.cached_property
    def prompt_token_length(self) -> int:
        return len(self.input_ids[0])

    @functools.cached_property
    def num_generated_tokens(self) -> int:
        return len(self.input_ids[1:])

    def expert_counts_by_layer(self) -> np.ndarray:
        expert_counts_by_layer = np.zeros(
            (self.num_layers, self.num_experts), dtype=np.int32
        )
        for layer_id, exp_ids in zip(self.layer_ids, self.expert_ids):
            exp_ids_flat = list(itertools.chain.from_iterable(exp_ids))
            expert_counts_by_layer[layer_id] += np.bincount(
                exp_ids_flat + list(range(self.num_experts))
            )

        return expert_counts_by_layer

    def without_prefix(self) -> "QueryTrace":
        return QueryTrace(
            layer_ids=self.layer_ids[self.num_layers :],
            expert_ids=self.expert_ids[self.num_layers :],
            input_ids=self.input_ids[self.num_layers :],
            has_prefix=False,
        )
