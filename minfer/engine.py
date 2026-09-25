import time
from dataclasses import dataclass

import torch
import torch.cuda.nvtx as nvtx

from minfer.block_manager import BlockAllocator
from minfer.config import EngineConfig, ModelConfig
from minfer.kernels.decode_attention import choose_num_splits
from minfer.model import AttentionMetadata, KVCache, LlamaModel, PrefillChunk
from minfer.sampling import SamplingParams, sample
from minfer.scheduler import Scheduler
from minfer.sequence import Sequence
from minfer.weights import load_state_dict, resolve_model_path


@dataclass
class RequestOutput:
    seq_id: int
    prompt_len: int
    output_ids: list[int]
    text: str | None
    ttft: float | None          # seconds from arrival to first generated token
    latency: float | None       # seconds from arrival to finish


class _DecodeGraph:
    """Static input buffers + captured CUDA graph for a pure-decode step of batch size bs."""

    def __init__(self, graph, logits):
        self.graph = graph
        self.logits = logits


class LLMEngine:
    def __init__(self, model: str, engine_config: EngineConfig | None = None, dtype=torch.bfloat16,
                 load_tokenizer: bool = True, verbose: bool = True):
        self.cfg = engine_config or EngineConfig()
        self.verbose = verbose
        path = resolve_model_path(model)
        self.model_cfg = ModelConfig.from_pretrained(path)
        self.cfg.max_model_len = min(self.cfg.max_model_len, self.model_cfg.max_position_embeddings)
        self.device = torch.device("cuda")
        self.model = LlamaModel(self.model_cfg, load_state_dict(path), self.device, dtype,
                                use_triton_rmsnorm=self.cfg.use_triton_rmsnorm,
                                use_triton_decode_attn=self.cfg.use_triton_decode_attn)
        torch.cuda.empty_cache()

        # size the paged KV cache from the memory left after weights, keeping headroom for activations
        free, total = torch.cuda.mem_get_info()
        activation_reserve = 768 * 2**20
        kv_bytes = int((free - activation_reserve) * self.cfg.kv_cache_memory_fraction)
        per_block = KVCache.bytes_per_block(self.model_cfg, self.cfg.block_size, dtype)
        num_blocks = kv_bytes // per_block
        self.model.kv_cache = KVCache(self.model_cfg, num_blocks, self.cfg.block_size, dtype, self.device)
        self.allocator = BlockAllocator(num_blocks, self.cfg.block_size)
        self.scheduler = Scheduler(self.cfg, self.allocator)
        self.max_blocks_per_seq = -(-self.cfg.max_model_len // self.cfg.block_size)
        if verbose:
            print(f"[minfer] KV cache: {num_blocks} blocks x {self.cfg.block_size} tokens "
                  f"= {num_blocks * self.cfg.block_size} tokens ({num_blocks * per_block / 2**30:.2f} GiB)")

        self.tokenizer = None
        if load_tokenizer:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.graphs: dict[int, _DecodeGraph] = {}
        if self.cfg.use_cuda_graphs:
            self._capture_cuda_graphs()

        self.num_steps = 0
        self.num_graph_steps = 0

    # ------------------------------------------------------------------ CUDA graphs
    def _capture_cuda_graphs(self):
        t0 = time.perf_counter()
        sizes = sorted(b for b in self.cfg.cuda_graph_batch_sizes if b <= self.cfg.max_num_seqs)
        max_bs = sizes[-1]
        dev = self.device
        self.g_input_ids = torch.zeros(max_bs, dtype=torch.long, device=dev)
        self.g_positions = torch.zeros(max_bs, dtype=torch.long, device=dev)
        self.g_slots = torch.zeros(max_bs, dtype=torch.long, device=dev)          # block 0 = scratch page
        self.g_block_tables = torch.zeros(max_bs, self.max_blocks_per_seq, dtype=torch.int32, device=dev)
        self.g_context_lens = torch.ones(max_bs, dtype=torch.int32, device=dev)
        pool = None
        for bs in reversed(sizes):  # largest first so smaller graphs reuse its memory pool
            meta = AttentionMetadata(
                slot_mapping=self.g_slots[:bs], num_decode=bs,
                block_tables=self.g_block_tables[:bs], context_lens=self.g_context_lens[:bs],
                max_decode_context=self.cfg.max_model_len,
                decode_num_splits=choose_num_splits(bs, self.model_cfg.num_kv_heads, self.cfg.max_model_len))
            args = (self.g_input_ids[:bs], self.g_positions[:bs], meta)
            self.model.forward(*args)  # warm-up: triton autotune/compile outside capture
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                logits = self.model.forward(*args)
            pool = g.pool()
            self.graphs[bs] = _DecodeGraph(g, logits)
        torch.cuda.synchronize()
        self.graph_sizes = sizes
        if self.verbose:
            print(f"[minfer] captured {len(sizes)} decode CUDA graphs (bs<={max_bs}) in {time.perf_counter() - t0:.1f}s")

    def _run_graph(self, input_ids, positions, slots, block_tables, context_lens) -> torch.Tensor:
        B = input_ids.shape[0]
        bs = next(s for s in self.graph_sizes if s >= B)
        self.g_input_ids[:B].copy_(input_ids, non_blocking=True)
        self.g_positions[:B].copy_(positions, non_blocking=True)
        self.g_slots[:B].copy_(slots, non_blocking=True)
        nb = block_tables.shape[1]
        self.g_block_tables[:B, :nb].copy_(block_tables, non_blocking=True)
        self.g_context_lens[:B].copy_(context_lens, non_blocking=True)
        if bs > B:  # padding rows: attend to 1 token in the scratch page, write to the scratch page
            self.g_slots[B:bs].zero_()
            self.g_block_tables[B:bs, 0] = 0
            self.g_context_lens[B:bs] = 1
        self.graphs[bs].graph.replay()
        return self.graphs[bs].logits[:B]

    # ------------------------------------------------------------------ requests
    def add_request(self, prompt: str | list[int], params: SamplingParams | None = None) -> Sequence:
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        params = params or SamplingParams()
        if len(ids) + params.max_tokens > self.cfg.max_model_len:
            params = SamplingParams(**{**params.__dict__, "max_tokens": self.cfg.max_model_len - len(ids)})
        seq = Sequence(ids, params)
        self.scheduler.add(seq)
        return seq

    # ------------------------------------------------------------------ one engine step
    def _prepare(self, batch: list[tuple[Sequence, int]]):
        bs_ = self.cfg.block_size
        dev = self.device
        # single-token work (decodes, or a 1-token prompt chunk) goes first, through the decode kernel
        batch = [x for x in batch if x[1] == 1] + [x for x in batch if x[1] > 1]
        input_ids, positions, slots = [], [], []
        dec_tables, dec_ctx = [], []
        chunks = []
        logits_idx, sample_seqs = [], []
        off = 0
        for seq, n in batch:
            nc = seq.num_computed
            bt = seq.block_table
            input_ids.extend(seq.token_ids[nc:nc + n])
            positions.extend(range(nc, nc + n))
            slots.extend(bt[p // bs_] * bs_ + p % bs_ for p in range(nc, nc + n))
            if n == 1:
                dec_tables.append(bt)
                dec_ctx.append(nc + 1)
            else:
                chunks.append((off, n, nc, bt))
            if nc + n == seq.num_tokens:
                logits_idx.append(off + n - 1)
                sample_seqs.append(seq)
            off += n

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=dev)
        positions = torch.tensor(positions, dtype=torch.long, device=dev)
        slots = torch.tensor(slots, dtype=torch.long, device=dev)
        meta = AttentionMetadata(slot_mapping=slots, num_decode=len(dec_ctx))
        if dec_ctx:
            nb = max(len(t) for t in dec_tables)
            meta.block_tables = torch.tensor([t + [0] * (nb - len(t)) for t in dec_tables],
                                             dtype=torch.int32, device=dev)
            meta.context_lens = torch.tensor(dec_ctx, dtype=torch.int32, device=dev)
            meta.max_decode_context = max(dec_ctx)
        for start, n, nc, bt in chunks:
            if nc == 0:
                meta.prefills.append(PrefillChunk(start, n, n, None, None))
                continue
            ctx = nc + n
            pos = torch.arange(ctx, device=dev)
            bt_t = torch.tensor(bt, dtype=torch.long, device=dev)
            ctx_slots = bt_t[pos // bs_] * bs_ + pos % bs_
            mask = pos[None, :] <= (nc + torch.arange(n, device=dev))[:, None]
            meta.prefills.append(PrefillChunk(start, n, ctx, ctx_slots, mask))
        return batch, input_ids, positions, meta, logits_idx, sample_seqs

    def step(self) -> list[Sequence]:
        """Run one scheduling + forward step. Returns sequences that finished this step."""
        nvtx.range_push("schedule")
        batch = self.scheduler.schedule()
        nvtx.range_pop()
        if not batch:
            return []
        nvtx.range_push("prepare")
        batch, input_ids, positions, meta, logits_idx, sample_seqs = self._prepare(batch)
        nvtx.range_pop()

        nvtx.range_push("forward")
        pure_decode = not meta.prefills
        if pure_decode and self.graphs and meta.num_decode <= self.graph_sizes[-1]:
            logits = self._run_graph(input_ids, positions, meta.slot_mapping, meta.block_tables, meta.context_lens)
            self.num_graph_steps += 1
        else:
            idx = None
            if not pure_decode:
                idx = torch.tensor(logits_idx, dtype=torch.long, device=self.device)
            logits = self.model.forward(input_ids, positions, meta, idx)
        nvtx.range_pop()

        nvtx.range_push("sample")
        if sample_seqs:
            temps = torch.tensor([s.params.temperature for s in sample_seqs], device=self.device)
            new_tokens = sample(logits, temps).tolist()
        else:
            new_tokens = []
        nvtx.range_pop()

        self.num_steps += 1
        for seq, n in batch:
            seq.num_computed += n
        now = time.perf_counter()
        finished = []
        eos = self.model_cfg.eos_token_id
        for seq, tok in zip(sample_seqs, new_tokens):
            seq.token_ids.append(tok)
            if seq.first_token_time is None:
                seq.first_token_time = now
            done = (seq.num_output_tokens >= seq.params.max_tokens
                    or seq.num_tokens >= self.cfg.max_model_len
                    or (tok == eos and not seq.params.ignore_eos))
            if done:
                seq.finish_time = now
                self.scheduler.finish(seq)
                finished.append(seq)
        return finished

    # ------------------------------------------------------------------ offline API
    def generate(self, prompts: list[str] | list[list[int]], params: SamplingParams | list[SamplingParams] | None = None,
                 decode_text: bool = True) -> list[RequestOutput]:
        if not isinstance(params, list):
            params = [params or SamplingParams()] * len(prompts)
        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, params)]
        while self.scheduler.has_work():
            self.step()
        outs = []
        for s in seqs:
            text = self.tokenizer.decode(s.output_ids, skip_special_tokens=True) if (decode_text and self.tokenizer) else None
            outs.append(RequestOutput(s.seq_id, s.prompt_len, s.output_ids, text,
                                      s.first_token_time - s.arrival_time, s.finish_time - s.arrival_time))
        return outs
