"""Continuous-batching scheduler with chunked prefill.

Every engine step the scheduler builds a fresh batch under a token budget
(max_num_batched_tokens):
  1. every running sequence that is decoding gets its 1 token (decodes go first so
     inter-token latency stays flat while long prompts are being ingested);
  2. running sequences that are part-way through their prompt get the next chunk;
  3. waiting requests are admitted, their prompts cut into chunks that fit the
     remaining budget.
Finished sequences leave the batch immediately and new ones join the next step,
so the batch never idles waiting for its longest member (unlike static batching).
When the KV cache runs out of pages, the most recently admitted sequence is
preempted: its pages are freed and it is re-queued to be recomputed later.
"""
from collections import deque

from minfer.block_manager import BlockAllocator
from minfer.config import EngineConfig
from minfer.sequence import Sequence, Status


class Scheduler:
    def __init__(self, cfg: EngineConfig, allocator: BlockAllocator):
        self.cfg = cfg
        self.allocator = allocator
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0
        # keep ~1% of pages free when admitting new work so running sequences can grow
        self.watermark = max(1, allocator.num_blocks // 100)

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _preempt(self, seq: Sequence):
        self.running.remove(seq)
        self.allocator.free(seq)
        seq.num_computed = 0
        seq.status = Status.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)

    def _reserve_or_preempt(self, seq: Sequence, n: int) -> bool:
        """Reserve pages for seq, evicting the newest running sequences if needed.
        Returns False if seq itself had to be evicted."""
        while not self.allocator.reserve(seq, n):
            victim = self.running[-1]
            self._preempt(victim)
            if victim is seq:
                return False
        return True

    def schedule(self) -> list[tuple[Sequence, int]]:
        budget = self.cfg.max_num_batched_tokens
        decodes, prefills = [], []
        preemptions_before = self.num_preemptions

        # 1. decodes
        for seq in list(self.running):
            if seq.status is not Status.RUNNING or seq.num_new_tokens != 1:
                continue
            if budget == 0:
                break
            if self._reserve_or_preempt(seq, 1):
                decodes.append((seq, 1))
                budget -= 1

        # 2. in-flight prefills
        for seq in list(self.running):
            if seq.status is not Status.RUNNING or seq.num_new_tokens <= 1 or budget == 0:
                continue
            n = min(seq.num_new_tokens, budget)
            if not self.cfg.enable_chunked_prefill and n < seq.num_new_tokens:
                continue
            if self._reserve_or_preempt(seq, n):
                prefills.append((seq, n))
                budget -= n

        # a sequence scheduled in pass 1 may have been evicted to make room in pass 2
        decodes = [(s, n) for s, n in decodes if s.status is Status.RUNNING]
        prefills = [(s, n) for s, n in prefills if s.status is Status.RUNNING]

        # 3. admit new requests (skip if we just had to evict: memory is tight)
        preempted_now = self.num_preemptions != preemptions_before
        while self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs and not preempted_now:
            seq = self.waiting[0]
            n = min(seq.num_new_tokens, budget)
            if not self.cfg.enable_chunked_prefill and n < seq.num_new_tokens:
                break
            if not self.allocator.reserve(seq, n, watermark=self.watermark):
                break
            self.waiting.popleft()
            seq.status = Status.RUNNING
            self.running.append(seq)
            prefills.append((seq, n))
            budget -= n

        return decodes + prefills

    def finish(self, seq: Sequence):
        seq.status = Status.FINISHED
        self.running.remove(seq)
        self.allocator.free(seq)
