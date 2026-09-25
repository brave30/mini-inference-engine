"""Llama-family decoder written from scratch, operating on a flattened ("varlen") batch.

All sequences scheduled in a step are concatenated into one token dimension T:
    [ decode tokens (1 per seq) | prefill chunk 0 | prefill chunk 1 | ... ]
Linear layers run once over all T tokens; only attention needs to know the sequence
boundaries, which it reads from AttentionMetadata.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from minfer.config import ModelConfig
from minfer.kernels.decode_attention import paged_decode_attention, paged_decode_attention_torch
from minfer.kernels.rmsnorm import add_rmsnorm_torch, fused_add_rmsnorm


@dataclass
class PrefillChunk:
    start: int                       # offset of the chunk in the flattened token dim
    length: int                      # tokens in this chunk
    context_len: int                 # tokens already in cache + this chunk
    ctx_slots: torch.Tensor | None   # cache slots for positions [0, context_len) (None if fresh prompt)
    mask: torch.Tensor | None        # [length, context_len] bool, True = attend


@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor                 # [T] int64 cache slot for every new token
    num_decode: int = 0
    block_tables: torch.Tensor | None = None   # [num_decode, max_blocks] int32
    context_lens: torch.Tensor | None = None   # [num_decode] int32
    max_decode_context: int = 0                # upper bound on context_lens
    decode_num_splits: int | None = None
    prefills: list[PrefillChunk] = field(default_factory=list)


class KVCache:
    """Paged KV cache: one [num_blocks, block_size, H_kv, D] tensor per layer for K and V."""

    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int, dtype, device):
        shape = (cfg.num_layers, 2, num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim)
        self.data = torch.zeros(shape, dtype=dtype, device=device)
        self.num_blocks = num_blocks
        self.block_size = block_size

    def layer(self, i: int):
        return self.data[i, 0], self.data[i, 1]

    @staticmethod
    def bytes_per_block(cfg: ModelConfig, block_size: int, dtype) -> int:
        return 2 * cfg.num_layers * block_size * cfg.num_kv_heads * cfg.head_dim * torch.finfo(dtype).bits // 8


def rope_cache(cfg: ModelConfig, max_len: int, device, dtype):
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.int64, device=device).float() / cfg.head_dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [T, heads, D]; cos/sin [T, D]  (HF "rotate_half" convention)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


class Layer:
    def __init__(self):
        self.input_norm = None
        self.post_attn_norm = None
        self.w_qkv = None
        self.b_qkv = None
        self.w_o = None
        self.w_gate_up = None
        self.w_down = None


class LlamaModel:
    def __init__(self, cfg: ModelConfig, state: dict[str, torch.Tensor], device="cuda", dtype=torch.bfloat16,
                 use_triton_rmsnorm=True, use_triton_decode_attn=True):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.use_triton_rmsnorm = use_triton_rmsnorm
        self.use_triton_decode_attn = use_triton_decode_attn
        self.scale = 1.0 / math.sqrt(cfg.head_dim)

        def w(name):
            return state.pop(name).to(device=device, dtype=dtype)

        self.embed = w("model.embed_tokens.weight")
        self.layers = []
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}."
            L = Layer()
            L.input_norm = w(p + "input_layernorm.weight")
            L.post_attn_norm = w(p + "post_attention_layernorm.weight")
            L.w_qkv = torch.cat([w(p + "self_attn.q_proj.weight"), w(p + "self_attn.k_proj.weight"),
                                 w(p + "self_attn.v_proj.weight")], 0)
            if cfg.attention_bias:
                L.b_qkv = torch.cat([w(p + "self_attn.q_proj.bias"), w(p + "self_attn.k_proj.bias"),
                                     w(p + "self_attn.v_proj.bias")], 0)
            L.w_o = w(p + "self_attn.o_proj.weight")
            L.w_gate_up = torch.cat([w(p + "mlp.gate_proj.weight"), w(p + "mlp.up_proj.weight")], 0)
            L.w_down = w(p + "mlp.down_proj.weight")
            self.layers.append(L)
        self.final_norm = w("model.norm.weight")
        self.lm_head = self.embed if cfg.tie_word_embeddings else w("lm_head.weight")
        self.cos, self.sin = rope_cache(cfg, cfg.max_position_embeddings, device, dtype)
        self.kv_cache: KVCache | None = None

    def _norm(self, x, residual, weight):
        if self.use_triton_rmsnorm:
            return fused_add_rmsnorm(x, residual, weight, self.cfg.rms_norm_eps)
        return add_rmsnorm_torch(x, residual, weight, self.cfg.rms_norm_eps)

    def _expand_kv(self, x):
        # [n, H_kv, D] -> [1, H, n, D]. Expanding explicitly keeps SDPA on its flash / mem-efficient
        # kernels; SDPA's enable_gqa=True path measured 10-23x slower for prefill on this setup.
        g = self.cfg.num_heads // self.cfg.num_kv_heads
        return x.transpose(0, 1).repeat_interleave(g, dim=0)[None]

    def _attention(self, q, k, v, layer_idx: int, meta: AttentionMetadata):
        cfg = self.cfg
        k_cache, v_cache = self.kv_cache.layer(layer_idx)
        flat_k = k_cache.view(-1, cfg.num_kv_heads, cfg.head_dim)
        flat_v = v_cache.view(-1, cfg.num_kv_heads, cfg.head_dim)
        flat_k.index_copy_(0, meta.slot_mapping, k)
        flat_v.index_copy_(0, meta.slot_mapping, v)

        out = torch.empty_like(q)
        nd = meta.num_decode
        if nd:
            if self.use_triton_decode_attn:
                paged_decode_attention(q[:nd], k_cache, v_cache, meta.block_tables, meta.context_lens,
                                       self.scale, meta.max_decode_context, out=out[:nd],
                                       num_splits=meta.decode_num_splits)
            else:
                out[:nd] = paged_decode_attention_torch(q[:nd], k_cache, v_cache, meta.block_tables,
                                                        meta.context_lens, self.scale, meta.max_decode_context)
        for c in meta.prefills:
            qs = q[c.start:c.start + c.length].transpose(0, 1).unsqueeze(0)       # [1, H, n, D]
            if c.ctx_slots is None:  # whole prompt in one chunk: attend to itself causally
                ks = k[c.start:c.start + c.length]
                vs = v[c.start:c.start + c.length]
                o = F.scaled_dot_product_attention(qs, self._expand_kv(ks), self._expand_kv(vs),
                                                   is_causal=True, scale=self.scale)
            else:  # later chunk: attend to everything cached so far (read back through the page table)
                ks = flat_k[c.ctx_slots]
                vs = flat_v[c.ctx_slots]
                o = F.scaled_dot_product_attention(qs, self._expand_kv(ks), self._expand_kv(vs),
                                                   attn_mask=c.mask, scale=self.scale)
            out[c.start:c.start + c.length] = o[0].transpose(0, 1)
        return out

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, meta: AttentionMetadata,
                logits_indices: torch.Tensor | None = None) -> torch.Tensor:
        cfg = self.cfg
        T = input_ids.shape[0]
        H, Hkv, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        cos, sin = self.cos[positions], self.sin[positions]
        h = F.embedding(input_ids, self.embed)
        residual = None
        for i, L in enumerate(self.layers):
            x, residual = self._norm(h, residual, L.input_norm)
            qkv = F.linear(x, L.w_qkv, L.b_qkv)
            q, k, v = qkv.split([H * D, Hkv * D, Hkv * D], dim=-1)
            q = apply_rope(q.view(T, H, D), cos, sin)
            k = apply_rope(k.view(T, Hkv, D), cos, sin)
            v = v.view(T, Hkv, D)
            attn = self._attention(q, k, v, i, meta)
            h = F.linear(attn.view(T, H * D), L.w_o)
            x, residual = self._norm(h, residual, L.post_attn_norm)
            gate, up = F.linear(x, L.w_gate_up).chunk(2, dim=-1)
            h = F.linear(F.silu(gate) * up, L.w_down)
        if logits_indices is not None:
            h, residual = h[logits_indices], residual[logits_indices]
        h, _ = self._norm(h, residual, self.final_norm)
        return F.linear(h, self.lm_head)
