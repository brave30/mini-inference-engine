# minfer: a mini LLM inference engine from scratch

A from-scratch LLM inference engine in PyTorch + Triton: paged KV cache, continuous-batching
scheduler with chunked prefill, custom Triton kernels (fused add+RMSNorm, paged split-K decode
attention) and CUDA-graph decode. It serves Llama-architecture models directly from their
safetensors. Hugging Face is used only for the tokenizer and as the baseline.

All numbers below were measured on **one NVIDIA GeForce RTX 5070 Laptop GPU (Blackwell sm_120,
8 GB, 36 SMs)**, Windows 11, PyTorch 2.11 + CUDA 12.8, triton-windows 3.8, transformers 5.17,
model **TinyLlama-1.1B-Chat-v1.0 in bf16** (22 layers, 32 query heads / 4 KV heads, head_dim 64).
Raw results are in `results/`.

## Headline results

**Throughput vs Hugging Face `generate`**, with both systems limited to the same N concurrent
sequences. 256 requests, output tok/s counting only requested tokens, EOS ignored on both sides
(`benchmarks/bench_throughput.py`):

| Concurrency N | Mixed lengths¹: HF | Mixed: minfer | **Speedup** | Fixed lengths²: HF | Fixed: minfer | **Speedup** |
|---|---|---|---|---|---|---|
| 8  | 122.1 | 715.6  | **5.86x** | 236.7 | 931.9  | **3.94x** |
| 16 | 174.9 | 1230.2 | **7.04x** | 386.5 | 1668.6 | **4.32x** |
| 32 | 281.6 | 2034.6 | **7.22x** | 479.0 | 2614.6 | **5.46x** |
| 64 | 306.7 | 2602.3 | **8.48x** | 527.1 | 3190.9 | **6.05x** |
| minfer uncapped (256) | | 4446.5 | 14.5x vs HF's best | | 5460.3 | 10.4x vs HF's best |

¹ prompt ~ U[128, 512], output ~ U[64, 512]. HF runs static batches of N that wait for their longest
member; minfer refills freed slots every step. ² every request is 256 prompt + 256 output tokens, so
HF wastes nothing on padding. This isolates engine efficiency (paged KV, fused kernels, CUDA graphs)
from the continuous-batching win. No preemptions occurred in any run; peak GPU memory was 5.6 GB.

| Kernel (vs PyTorch eager) | Speedup | Details |
|---|---|---|
| Fused residual-add + RMSNorm (Triton) | **2.7x to 5.2x** (4.8x at 32 rows) | 264 GB/s at 8192 rows; matches `torch.compile` |
| Paged decode attention, split-K + GQA (Triton) | **5.0x to 8.8x** (5.0x at bs=32, ctx 512) | up to 253 GB/s of KV read |

| Decode latency per token step (512-token context) | Triton, no graphs | Triton + CUDA graphs | Reduction |
|---|---|---|---|
| batch 1 | 28.07 ms | **9.60 ms** | **-65.8%** |
| batch 8 | 29.10 ms | **10.02 ms** | **-65.6%** |
| batch 32 | 26.25 ms | **11.94 ms** | **-54.5%** |
| batch 64 | 28.50 ms | **17.55 ms** | **-38.4%** |

Nsight Systems (bs=32 decode, 50 steps): eager launch issues **470 kernel launches per step**,
keeping the GPU only **28%** busy (9.97 ms of kernels in a 35.45 ms step). CUDA graphs replace them
with 5 launch calls per step: same kernel time (10.18 ms), step time **12.55 ms (-64.6%)**, GPU
**81%** busy.

## Architecture

```
minfer/
  engine.py          LLMEngine: step loop, metadata prep, CUDA-graph capture/replay, sampling
  scheduler.py       continuous batching + chunked prefill + preemption (recompute)
  block_manager.py   paged KV allocator (16-token pages, free list, per-sequence block tables)
  model.py           Llama decoder over a flattened varlen batch; paged KV cache tensor
  kernels/rmsnorm.py            Triton fused residual-add + RMSNorm
  kernels/decode_attention.py   Triton paged decode attention (GQA-grouped, split-K + LSE merge)
  sampling.py        greedy / temperature sampling (no host sync)
benchmarks/          throughput vs HF, kernel microbenchmarks, decode latency, Nsight driver
tests/               kernel tests vs PyTorch; end-to-end token match vs HF generate
```

**Paged KV cache.** The cache is one `[layers, 2, num_blocks, 16, kv_heads, head_dim]` tensor sized
from the free memory left after the weights load (159,072 token slots = 3.3 GiB here). Sequences get
16-token pages on demand through a block table, so memory is committed per page actually used, not
per `max_model_len`. Waste is at most 15 slots per sequence. Page 0 is a scratch page that CUDA-graph
padding rows write into.

**Continuous batching + chunked prefill.** Each step the scheduler builds a fresh batch under a
2048-token budget. It schedules every decoding sequence first (1 token each), then continues
in-flight prompts, then admits new requests, cutting their prompts into chunks that fit the
remaining budget. Finished sequences leave immediately and new ones join the next step. Long prompts
never stall decodes, and the batch never waits for its longest member. When pages run out, the
newest sequence is preempted (pages freed, recomputed later).

**One forward over mixed work.** Decode tokens and prefill chunks are concatenated into one token
dimension. The QKV/O/MLP GEMMs (with fused QKV and gate/up weights) run once over all of them.
Only attention splits: decode tokens go through the Triton paged kernel, and prefill chunks go
through SDPA. A later chunk reads its earlier context back through the page table.

**Decode attention kernel.** Grid `(batch, kv_head, split)`. Each program processes all 8 query heads
of a GQA group with `tl.dot`, so every K/V page is read from HBM once per group, not once per head.
It uses an online softmax in fp32 with `exp2`. When `batch x kv_heads` is too small to fill the 36
SMs, the context is split across programs (flash-decoding) and a second kernel merges the partials by
log-sum-exp. That's why batch 1 with a 2048-token context still runs in 25 us.

**CUDA graphs.** Pure-decode steps for batch sizes 1 to 256 (15 buckets) are captured once at
startup with a shared memory pool, over static input buffers. At runtime the batch is padded to the
next bucket (padding rows attend to the 1-token scratch page) and the graph is replayed. Mixed
prefill steps run eagerly.

## Correctness

* `tests/test_kernels.py`: both Triton kernels vs PyTorch references (ragged context lengths,
  randomly scattered pages, 1 to 8 splits, 896/2048/4096 hidden sizes).
* `tests/test_correctness.py`: greedy generation vs HF `generate` on 12 prompts, under 4 engine
  configs: default, 48-token chunked prefill, all-PyTorch, and a tiny KV cache that forces
  preemption. 77 to 86% of generated tokens are identical to HF. For every sequence that diverges, the
  test runs HF on the shared prefix and checks that minfer's token is within 2 bf16 ULPs of HF's top
  logit. Every divergence is an exact or 1-ULP tie (measured logit gaps of 0.0 and 0.0625), i.e.
  bf16 reduction-order noise, not an engine bug.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate      # (source .venv/bin/activate on Linux)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install triton-windows transformers safetensors huggingface_hub numpy   # 'triton' on Linux

python tests/test_kernels.py
python tests/test_correctness.py
python benchmarks/bench_kernels.py        # -> results/kernels.json
python benchmarks/bench_latency.py        # -> results/latency.json
python benchmarks/bench_throughput.py     # -> results/throughput.json (HF runs take a while)

# Nsight Systems
nsys profile -t cuda,nvtx -o results/nsys_decode_eager python benchmarks/profile_decode.py --graphs 0
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o results/nsys_decode_graphs python benchmarks/profile_decode.py --graphs 1
nsys stats --force-export true -q -r nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum --filter-nvtx decode_steps \
    --format csv results/nsys_decode_eager.nsys-rep > results/nsys_eager.csv    # same for graphs
python benchmarks/nsys_summary.py
```

```python
from minfer import LLMEngine, SamplingParams
llm = LLMEngine("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(llm.generate(["The capital of France is"], SamplingParams(max_tokens=32))[0].text)
```

## Notes on measurement

* **Baselines are made strong on purpose.** The eager decode-attention baseline uses one flat
  `index_select` gather plus GQA-grouped matmuls. A naive einsum/`repeat_interleave` version was
  15 to 85x slower than Triton and would have inflated the speedup. SDPA with `enable_gqa=True` was
  7 to 23x slower than expanding heads on this setup, so neither minfer's prefill nor the eager
  baseline uses it.
* The RMSNorm speedup is vs eager PyTorch. `torch.compile` generates an equally fast fused
  kernel (within ~15% either way), so the win is fusion itself, not something Inductor can't do.
* This is a 53 W laptop GPU on Windows (WDDM), where kernel-launch overhead is high. That is why
  CUDA graphs help so much here. On a datacenter GPU on Linux the graph win at bs=1 is typically smaller.
