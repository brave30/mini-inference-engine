"""Kernel microbenchmarks: Triton vs PyTorch eager (and torch.compile where relevant).

Timings use triton.testing.do_bench (CUDA events, L2 flushed between reps, median).
Shapes are TinyLlama-1.1B: hidden 2048, 32 query heads / 4 KV heads, head_dim 64,
16-token KV pages.

Usage: python benchmarks/bench_kernels.py   -> results/kernels.json
"""
import json
import math
import os
import sys

import torch
import triton

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from minfer.kernels.decode_attention import paged_decode_attention, paged_decode_attention_torch
from minfer.kernels.rmsnorm import add_rmsnorm_torch, fused_add_rmsnorm
from test_kernels import _make_paged


def bench(fn):
    return triton.testing.do_bench(fn, warmup=50, rep=300, return_mode="median")


def bench_rmsnorm():
    rows_out = []
    w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    compiled = torch.compile(add_rmsnorm_torch)
    for rows in (1, 32, 256, 2048, 8192):
        x = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
        r = torch.randn_like(x)
        t_eager = bench(lambda: add_rmsnorm_torch(x, r, w, 1e-5))
        t_tri = bench(lambda: fused_add_rmsnorm(x, r, w, 1e-5))
        try:
            t_comp = bench(lambda: compiled(x, r, w, 1e-5))
        except Exception as e:  # torch.compile/inductor may be unavailable on some Windows setups
            print("torch.compile unavailable:", type(e).__name__)
            t_comp = None
        # bytes moved by the fused kernel: read x, r, w; write y, r
        gbps = (4 * rows * 2048 * 2 + 2048 * 2) / (t_tri * 1e-3) / 1e9
        rows_out.append(dict(rows=rows, eager_us=t_eager * 1e3, triton_us=t_tri * 1e3,
                             compile_us=None if t_comp is None else t_comp * 1e3,
                             speedup_vs_eager=t_eager / t_tri, triton_gbps=gbps))
        comp = "" if t_comp is None else f"  torch.compile {t_comp * 1e3:7.1f}us"
        print(f"rmsnorm rows={rows:5d}  eager {t_eager * 1e3:7.1f}us  triton {t_tri * 1e3:7.1f}us{comp}  "
              f"speedup {t_eager / t_tri:5.2f}x  ({gbps:5.0f} GB/s)", flush=True)
    return rows_out


def bench_decode_attention():
    rows_out = []
    H, Hkv, D, page = 32, 4, 64, 16
    scale = 1 / math.sqrt(D)
    for B, ctx in [(1, 512), (1, 2048), (8, 512), (8, 2048), (32, 512), (32, 1024), (64, 512), (128, 512)]:
        k, v, bt, cl = _make_paged([ctx] * B, Hkv, D, page, num_blocks=B * ctx // page + 8)
        q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
        t_tri = bench(lambda: paged_decode_attention(q, k, v, bt, cl, scale, ctx))
        t_eager = bench(lambda: paged_decode_attention_torch(q, k, v, bt, cl, scale, ctx))
        t_sdpa = bench(lambda: paged_decode_attention_torch(q, k, v, bt, cl, scale, ctx, use_sdpa=True))
        kv_bytes = 2 * B * ctx * Hkv * D * 2
        gbps = kv_bytes / (t_tri * 1e-3) / 1e9
        rows_out.append(dict(batch=B, context=ctx, triton_us=t_tri * 1e3, eager_us=t_eager * 1e3,
                             eager_sdpa_us=t_sdpa * 1e3, speedup_vs_eager=t_eager / t_tri,
                             speedup_vs_eager_sdpa=t_sdpa / t_tri, triton_kv_gbps=gbps))
        print(f"decode-attn B={B:3d} ctx={ctx:5d}  eager {t_eager * 1e3:8.1f}us  eager+SDPA {t_sdpa * 1e3:8.1f}us  "
              f"triton {t_tri * 1e3:7.1f}us  speedup {t_eager / t_tri:5.1f}x / {t_sdpa / t_tri:5.1f}x  "
              f"({gbps:5.0f} GB/s KV)", flush=True)
    return rows_out


if __name__ == "__main__":
    res = {"gpu": torch.cuda.get_device_name(0), "rmsnorm": bench_rmsnorm(), "decode_attention": bench_decode_attention()}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "kernels.json"), "w") as f:
        json.dump(res, f, indent=1)
