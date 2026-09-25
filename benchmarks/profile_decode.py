"""Workload for Nsight Systems: steady-state decode at a fixed batch size.

    nsys profile -t cuda,nvtx -o results/nsys_decode_graphs python benchmarks/profile_decode.py --graphs 1
    nsys profile -t cuda,nvtx -o results/nsys_decode_eager  python benchmarks/profile_decode.py --graphs 0
    nsys stats -r nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum results/nsys_decode_eager.nsys-rep

Every engine step is wrapped in NVTX ranges (schedule / prepare / forward / sample, see
engine.py); the measured region is additionally wrapped in a "decode_steps" range.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from minfer import LLMEngine, SamplingParams
from minfer.config import EngineConfig

ap = argparse.ArgumentParser()
ap.add_argument("--graphs", type=int, default=1)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--prompt", type=int, default=512)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--torch-attn", action="store_true", help="use the PyTorch eager attention + RMSNorm")
args = ap.parse_args()

cfg = EngineConfig(use_cuda_graphs=bool(args.graphs), use_triton_decode_attn=not args.torch_attn,
                   use_triton_rmsnorm=not args.torch_attn)
eng = LLMEngine("TinyLlama/TinyLlama-1.1B-Chat-v1.0", cfg, load_tokenizer=False, verbose=False)
for i in range(args.batch):
    eng.add_request([1] + [1000 + (i * 7 + j) % 30000 for j in range(args.prompt - 1)],
                    SamplingParams(max_tokens=args.steps + 30, ignore_eos=True))
while any(s.num_new_tokens > 1 for s in eng.scheduler.running) or eng.scheduler.waiting:
    eng.step()
for _ in range(20):
    eng.step()
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("decode_steps")
for _ in range(args.steps):
    eng.step()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
