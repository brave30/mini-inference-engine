"""Per-token decode latency: CUDA graphs vs eager launch, and Triton kernels vs PyTorch.

B sequences with a `--prompt`-token prompt are prefetched, then we time pure decode
steps (engine.step: schedule + prepare + forward + sample + bookkeeping), i.e. the
real inter-token latency a user sees. Median over `--steps` steps.

Usage: python benchmarks/bench_latency.py   -> results/latency.json
"""
import argparse
import gc
import json
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from minfer import LLMEngine, SamplingParams
from minfer.config import EngineConfig

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

VARIANTS = {
    "pytorch eager (no triton, no graphs)": dict(use_triton_rmsnorm=False, use_triton_decode_attn=False, use_cuda_graphs=False),
    "triton kernels, no graphs": dict(use_cuda_graphs=False),
    "triton kernels + cuda graphs": dict(),
}


def measure(eng: LLMEngine, batch: int, prompt: int, steps: int) -> float:
    ids = [[1] + [1000 + (i * 7 + j) % 30000 for j in range(prompt - 1)] for i in range(batch)]
    for p in ids:
        eng.add_request(p, SamplingParams(max_tokens=steps + 40, ignore_eos=True))
    while any(s.num_new_tokens > 1 for s in eng.scheduler.running) or eng.scheduler.waiting:
        eng.step()  # prefill
    for _ in range(10):
        eng.step()  # warm-up decodes
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eng.step()  # step() ends with a .tolist() on the sampled tokens, so it is synchronous
        times.append(time.perf_counter() - t0)
    while eng.scheduler.has_work():
        eng.step()
    return statistics.median(times) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()
    res = {}
    for name, ov in VARIANTS.items():
        eng = LLMEngine(MODEL, EngineConfig(**ov), load_tokenizer=False, verbose=False)
        res[name] = {}
        for b in args.batches:
            ms = measure(eng, b, args.prompt, args.steps)
            res[name][b] = ms
            print(f"{name:38s} bs={b:3d}  {ms:7.2f} ms/token-step  ({b / ms * 1e3:8.1f} tok/s)", flush=True)
        del eng
        gc.collect()
        torch.cuda.empty_cache()
    base, trit, graph = (res[k] for k in VARIANTS)
    print()
    for b in args.batches:
        print(f"bs={b:3d}: cuda graphs cut latency {100 * (1 - graph[b] / trit[b]):5.1f}% vs triton-eager, "
              f"{100 * (1 - graph[b] / base[b]):5.1f}% vs pytorch-eager")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "latency.json"), "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "prompt_len": args.prompt, "ms_per_step": res}, f, indent=1)


if __name__ == "__main__":
    main()
