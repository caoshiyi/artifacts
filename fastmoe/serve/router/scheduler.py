import random
from collections import defaultdict


class Scheduler:
    def __init__(
        self,
        max_running_seq,
        max_prefill_num_token,
        max_total_num_token,
    ):
        self.max_running_seq = max_running_seq
        self.max_prefill_num_token = max_prefill_num_token
        self.max_total_num_token = max_total_num_token

    def get_priority_queue(self, forward_queue):
        return forward_queue