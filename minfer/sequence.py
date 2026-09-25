import enum
import itertools
import time

from minfer.sampling import SamplingParams

_ids = itertools.count()


class Status(enum.Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2


class Sequence:
    def __init__(self, prompt_ids: list[int], params: SamplingParams):
        self.seq_id = next(_ids)
        self.token_ids = list(prompt_ids)
        self.prompt_len = len(prompt_ids)
        self.params = params
        self.num_computed = 0            # tokens whose K/V are already in the cache
        self.block_table: list[int] = []
        self.status = Status.WAITING
        self.num_preemptions = 0
        self.arrival_time = time.perf_counter()
        self.first_token_time: float | None = None
        self.finish_time: float | None = None

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_new_tokens(self) -> int:
        return self.num_tokens - self.num_computed

    @property
    def output_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len:]

    @property
    def num_output_tokens(self) -> int:
        return self.num_tokens - self.prompt_len
