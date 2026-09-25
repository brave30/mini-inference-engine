"""Paged KV-cache allocator.

The cache is carved into fixed-size physical blocks of `block_size` tokens. Each
sequence owns a block table (logical block i -> physical block id), grown one block
at a time as it produces tokens, so memory is committed per 16 tokens actually used
instead of reserving max_model_len per sequence up front. Fragmentation is bounded
by at most block_size - 1 wasted slots per sequence.
"""
from collections import deque

from minfer.sequence import Sequence


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int, reserved: int = 1):
        # block 0 is reserved as a scratch page for CUDA-graph padding slots
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks = deque(range(reserved, num_blocks))

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    def blocks_needed(self, seq: Sequence, num_new_tokens: int) -> int:
        total = seq.num_computed + num_new_tokens
        return max(0, -(-total // self.block_size) - len(seq.block_table))

    def can_allocate(self, n: int, watermark: int = 0) -> bool:
        return len(self.free_blocks) - n >= watermark

    def reserve(self, seq: Sequence, num_new_tokens: int, watermark: int = 0) -> bool:
        """Make sure seq has pages for its next num_new_tokens tokens."""
        need = self.blocks_needed(seq, num_new_tokens)
        if need == 0:
            return True
        if not self.can_allocate(need, watermark):
            return False
        seq.block_table.extend(self.free_blocks.popleft() for _ in range(need))
        return True

    def free(self, seq: Sequence):
        self.free_blocks.extend(seq.block_table)
        seq.block_table = []
