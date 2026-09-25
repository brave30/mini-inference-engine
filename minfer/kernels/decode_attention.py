"""Paged decode attention (split-K / flash-decoding style) in Triton.

Layout
  q            [B, H, D]                     one query token per sequence
  k/v cache    [num_blocks, PAGE, H_kv, D]   one layer's paged cache
  block_tables [B, max_blocks] int32         logical page -> physical page
  context_lens [B] int32                     tokens in cache (incl. the current one)

Grid (B, H_kv, num_splits). Each program handles every query head of one GQA group
against one contiguous partition of the context, so each K/V page is read from HBM
once per group instead of once per query head. With few sequences the context is
split across programs (flash-decoding) and a second kernel merges the partial results
by their log-sum-exp, which keeps all SMs busy even at batch size 1.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_attn_kernel(
    Q, K, V, BlockTables, CtxLens, O, OPart, LsePart,
    sm_scale,
    stride_qb, stride_qh,
    stride_kblk, stride_kslot, stride_kh,
    stride_bt,
    stride_ob, stride_oh,
    stride_pb, stride_ph, stride_ps,
    stride_lb, stride_lh,
    GROUP: tl.constexpr, GROUP_PAD: tl.constexpr, HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr, BLOCK_N: tl.constexpr, PARTITION: tl.constexpr,
    SPLIT: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    s = tl.program_id(2)

    ctx = tl.load(CtxLens + b)
    start = s * PARTITION
    end = tl.minimum(start + PARTITION, ctx)

    offs_g = tl.arange(0, GROUP_PAD)
    offs_d = tl.arange(0, HEAD_DIM)
    gmask = offs_g < GROUP
    heads = kvh * GROUP + offs_g

    q = tl.load(Q + b * stride_qb + heads[:, None] * stride_qh + offs_d[None, :],
                mask=gmask[:, None], other=0.0)

    m_i = tl.full([GROUP_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([GROUP_PAD], dtype=tl.float32)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634  # fold log2(e) so we can use exp2

    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < end
        page = tl.load(BlockTables + b * stride_bt + offs_n // PAGE, mask=nmask, other=0)
        kv_off = page.to(tl.int64) * stride_kblk + (offs_n % PAGE) * stride_kslot + kvh * stride_kh
        k = tl.load(K + kv_off[:, None] + offs_d[None, :], mask=nmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where(nmask[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_new = tl.where(m_new == float("-inf"), 0.0, m_new)  # padded GQA rows stay finite
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(V + kv_off[:, None] + offs_d[None, :], mask=nmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    if SPLIT:
        # partition results + base-2 log-sum-exp; empty partitions get -inf weight
        lse = tl.where(l_i > 0, m_i + tl.log2(l_safe), float("-inf"))
        tl.store(OPart + b * stride_pb + heads[:, None] * stride_ph + s * stride_ps + offs_d[None, :],
                 out, mask=gmask[:, None])
        tl.store(LsePart + b * stride_lb + heads * stride_lh + s, lse, mask=gmask)
    else:
        tl.store(O + b * stride_ob + heads[:, None] * stride_oh + offs_d[None, :],
                 out.to(O.dtype.element_ty), mask=gmask[:, None])


@triton.jit
def _merge_partitions_kernel(
    O, OPart, LsePart,
    stride_ob, stride_oh, stride_pb, stride_ph, stride_ps, stride_lb, stride_lh,
    HEAD_DIM: tl.constexpr, NUM_SPLITS: tl.constexpr, SPLITS_PAD: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, SPLITS_PAD)
    smask = offs_s < NUM_SPLITS
    lse = tl.load(LsePart + b * stride_lb + h * stride_lh + offs_s, mask=smask, other=float("-inf"))
    m = tl.max(lse, axis=0)
    w = tl.where(lse > float("-inf"), tl.exp2(lse - m), 0.0)
    w = w / tl.sum(w, axis=0)
    offs_d = tl.arange(0, HEAD_DIM)
    o = tl.load(OPart + b * stride_pb + h * stride_ph + offs_s[:, None] * stride_ps + offs_d[None, :],
                mask=(w > 0)[:, None], other=0.0)
    out = tl.sum(o * w[:, None], axis=0)
    tl.store(O + b * stride_ob + h * stride_oh + offs_d, out.to(O.dtype.element_ty))


_NUM_SMS = None
BLOCK_N = 64


def _num_sms() -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    return _NUM_SMS


def choose_num_splits(batch: int, num_kv_heads: int, max_context: int) -> int:
    """Split the context only when (batch x kv_heads) programs can't fill the GPU,
    and never make a partition shorter than 256 tokens."""
    programs = batch * num_kv_heads
    if programs >= 2 * _num_sms():
        return 1
    by_occupancy = triton.cdiv(2 * _num_sms(), programs)
    by_length = max(1, max_context // 256)
    return max(1, min(by_occupancy, by_length))


def paged_decode_attention(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    block_tables: torch.Tensor, context_lens: torch.Tensor, sm_scale: float,
    max_context: int, out: torch.Tensor | None = None, num_splits: int | None = None,
) -> torch.Tensor:
    """max_context is an upper bound on context_lens (it only sizes the split partitions,
    so a static bound can be used under CUDA-graph capture)."""
    B, H, D = q.shape
    _, page, H_kv, _ = k_cache.shape
    group = H // H_kv
    if out is None:
        out = torch.empty_like(q)
    if num_splits is None:
        num_splits = choose_num_splits(B, H_kv, max_context)
    partition = triton.cdiv(triton.cdiv(max_context, num_splits), BLOCK_N) * BLOCK_N
    num_splits = triton.cdiv(max_context, partition)
    group_pad = max(16, triton.next_power_of_2(group))  # tl.dot needs M >= 16
    split = num_splits > 1
    if split:
        o_part = torch.empty(B, H, num_splits, D, dtype=torch.float32, device=q.device)
        lse_part = torch.empty(B, H, num_splits, dtype=torch.float32, device=q.device)
        sp = (o_part.stride(0), o_part.stride(1), o_part.stride(2), lse_part.stride(0), lse_part.stride(1))
    else:
        o_part, lse_part = out, out  # unused
        sp = (0, 0, 0, 0, 0)
    _paged_decode_attn_kernel[(B, H_kv, num_splits)](
        q, k_cache, v_cache, block_tables, context_lens, out, o_part, lse_part,
        sm_scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        block_tables.stride(0),
        out.stride(0), out.stride(1),
        *sp,
        GROUP=group, GROUP_PAD=group_pad, HEAD_DIM=D, PAGE=page, BLOCK_N=BLOCK_N,
        PARTITION=partition, SPLIT=split, num_warps=4, num_stages=2,
    )
    if split:
        _merge_partitions_kernel[(B, H)](
            out, o_part, lse_part,
            out.stride(0), out.stride(1), *sp,
            HEAD_DIM=D, NUM_SPLITS=num_splits, SPLITS_PAD=triton.next_power_of_2(num_splits),
            num_warps=1,
        )
    return out


def paged_decode_attention_torch(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    block_tables: torch.Tensor, context_lens: torch.Tensor, sm_scale: float,
    max_context: int, use_sdpa: bool = False,
) -> torch.Tensor:
    """PyTorch eager baseline: gather pages into a contiguous padded KV tensor, then attend.

    Written to be a strong baseline: one flat index_select per K/V, GQA handled by grouping
    query heads (no K/V repeat), bf16 matmuls with an fp32 softmax. `use_sdpa` instead calls
    SDPA on the gathered KV with the heads expanded (SDPA's enable_gqa path is ~7x slower here)."""
    B, H, D = q.shape
    _, page, H_kv, _ = k_cache.shape
    group = H // H_kv
    pos = torch.arange(max_context, device=q.device)
    slots = (block_tables[:, pos // page].long() * page + pos % page).view(-1)
    k = k_cache.view(-1, H_kv, D).index_select(0, slots).view(B, max_context, H_kv, D).transpose(1, 2)
    v = v_cache.view(-1, H_kv, D).index_select(0, slots).view(B, max_context, H_kv, D).transpose(1, 2)
    valid = (pos[None, :] < context_lens[:, None])[:, None, None, :]   # [B, 1, 1, L]
    if use_sdpa:
        out = torch.nn.functional.scaled_dot_product_attention(
            q[:, :, None, :], k.repeat_interleave(group, 1), v.repeat_interleave(group, 1),
            attn_mask=valid, scale=sm_scale)
        return out[:, :, 0, :]
    qg = q.view(B, H_kv, group, D)
    scores = torch.matmul(qg, k.transpose(-1, -2)).float() * sm_scale   # [B, H_kv, G, L]
    scores = scores.masked_fill(~valid, float("-inf"))
    p = torch.softmax(scores, dim=-1).to(q.dtype)
    return torch.matmul(p, v).view(B, H, D)
