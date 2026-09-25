"""Correctness tests for the Triton kernels against PyTorch references.
Run: python -m pytest tests/test_kernels.py -q   (or python tests/test_kernels.py)"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minfer.kernels.decode_attention import paged_decode_attention, paged_decode_attention_torch
from minfer.kernels.rmsnorm import add_rmsnorm_torch, fused_add_rmsnorm


def test_rmsnorm():
    torch.manual_seed(0)
    for rows in (1, 7, 256, 2048):
        for n in (2048, 4096, 896):
            x = torch.randn(rows, n, device="cuda", dtype=torch.bfloat16)
            r = torch.randn_like(x)
            w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
            y_ref, r_ref = add_rmsnorm_torch(x, r.clone(), w, 1e-5)
            r_tri = r.clone()
            y, r_out = fused_add_rmsnorm(x, r_tri, w, 1e-5)
            assert torch.equal(r_out, r_ref)
            torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=1e-2)
            y2, _ = fused_add_rmsnorm(x, None, w, 1e-5)
            y2_ref, _ = add_rmsnorm_torch(x, None, w, 1e-5)
            torch.testing.assert_close(y2, y2_ref, atol=2e-2, rtol=1e-2)


def _make_paged(ctx_lens, H_kv, D, page, num_blocks, dtype=torch.bfloat16):
    k_cache = torch.randn(num_blocks, page, H_kv, D, device="cuda", dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    max_blocks = max(math.ceil(c / page) for c in ctx_lens)
    perm = torch.randperm(num_blocks - 1, device="cuda") + 1  # scattered physical pages
    bt = torch.zeros(len(ctx_lens), max_blocks, dtype=torch.int32, device="cuda")
    i = 0
    for b, c in enumerate(ctx_lens):
        n = math.ceil(c / page)
        bt[b, :n] = perm[i:i + n]
        i += n
    return k_cache, v_cache, bt, torch.tensor(ctx_lens, dtype=torch.int32, device="cuda")


def test_decode_attention():
    torch.manual_seed(0)
    H, H_kv, D, page = 32, 4, 64, 16
    cases = [[1], [17], [1000], [2048], [5, 300, 1, 2047, 64, 65], list(range(1, 200, 7))]
    for ctx in cases:
        k, v, bt, cl = _make_paged(ctx, H_kv, D, page, num_blocks=4096)
        q = torch.randn(len(ctx), H, D, device="cuda", dtype=torch.bfloat16)
        scale = 1 / math.sqrt(D)
        ref = paged_decode_attention_torch(q, k, v, bt, cl, scale, max(ctx))
        for splits in (None, 1, 3, 8):
            out = paged_decode_attention(q, k, v, bt, cl, scale, max(ctx), num_splits=splits)
            torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        # a looser static bound (as used under CUDA graphs) must give the same answer
        out = paged_decode_attention(q, k, v, bt, cl, scale, 2048)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    test_rmsnorm()
    print("rmsnorm ok")
    test_decode_attention()
    print("decode attention ok")
