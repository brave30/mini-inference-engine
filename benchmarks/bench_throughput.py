"""Offline throughput: minfer vs Hugging Face `generate`.

Both systems get the same requests and the same concurrency limit N:
  * HF: static batches of N requests (left-padded), each batch runs until its longest
    request is done (max_new_tokens = longest in batch; shorter ones are padding work).
  * minfer: continuous batching with max_num_seqs = N, paged KV, chunked prefill.
Throughput = requested output tokens / wall-clock seconds (only useful tokens count).
EOS is ignored on both sides so every request produces exactly its target length.

Workloads
  mixed : 256 requests, prompt len ~ U[128, 512], output len ~ U[64, 512]
  fixed : 256 requests, prompt len 256, output len 256  (no length variance: isolates
          raw engine efficiency from the continuous-batching advantage)

Usage
  python benchmarks/bench_throughput.py                 # full sweep, writes results/throughput.json
  python benchmarks/bench_throughput.py --system minfer --batch 32 --workload mixed
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def make_workload(kind: str, n: int = 256, seed: int = 0, vocab: int = 32000):
    rng = random.Random(seed)
    reqs = []
    for _ in range(n):
        if kind == "mixed":
            p, o = rng.randint(128, 512), rng.randint(64, 512)
        else:
            p, o = 256, 256
        ids = [1] + [rng.randint(100, vocab - 1) for _ in range(p - 1)]
        reqs.append((ids, o))
    return reqs


def run_hf(reqs, batch: int) -> dict:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    pad = 2

    def run_batch(chunk):
        maxp = max(len(ids) for ids, _ in chunk)
        maxo = max(o for _, o in chunk)
        input_ids = torch.full((len(chunk), maxp), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxp), dtype=torch.long)
        for i, (ids, _) in enumerate(chunk):
            input_ids[i, maxp - len(ids):] = torch.tensor(ids)
            attn[i, maxp - len(ids):] = 1
        model.generate(input_ids=input_ids.cuda(), attention_mask=attn.cuda(), max_new_tokens=maxo,
                       min_new_tokens=maxo, do_sample=False, pad_token_id=pad)

    run_batch(reqs[:batch][:4])  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(reqs), batch):
        run_batch(reqs[i:i + batch])
    torch.cuda.synchronize()
    return {"seconds": time.perf_counter() - t0, "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30}


def run_minfer(reqs, batch: int, **overrides) -> dict:
    from minfer import LLMEngine, SamplingParams
    from minfer.config import EngineConfig
    cfg = EngineConfig(max_num_seqs=batch, **overrides)
    eng = LLMEngine(MODEL, cfg, load_tokenizer=False, verbose=False)
    eng.generate([ids for ids, _ in reqs[:4]], SamplingParams(max_tokens=16, ignore_eos=True), decode_text=False)
    eng.num_steps = eng.num_graph_steps = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = eng.generate([ids for ids, _ in reqs],
                        [SamplingParams(max_tokens=o, ignore_eos=True) for _, o in reqs], decode_text=False)
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    assert all(len(o.output_ids) == n for o, (_, n) in zip(outs, reqs))
    ttft = sorted(o.ttft for o in outs)
    return {"seconds": secs, "steps": eng.num_steps, "graph_steps": eng.num_graph_steps,
            "preemptions": eng.scheduler.num_preemptions,
            "kv_cache_tokens": eng.allocator.num_blocks * eng.cfg.block_size,
            "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30}


def single(args):
    reqs = make_workload(args.workload, args.num_requests)
    out_tokens = sum(o for _, o in reqs)
    in_tokens = sum(len(i) for i, _ in reqs)
    if args.system == "hf":
        r = run_hf(reqs, args.batch)
    else:
        overrides = json.loads(args.overrides) if args.overrides else {}
        r = run_minfer(reqs, args.batch, **overrides)
    r.update(system=args.system, batch=args.batch, workload=args.workload, num_requests=len(reqs),
             output_tokens=out_tokens, prompt_tokens=in_tokens,
             output_tok_per_s=out_tokens / r["seconds"], total_tok_per_s=(out_tokens + in_tokens) / r["seconds"])
    print("RESULT " + json.dumps(r))


def sub(system, batch, workload, n, overrides=None):
    cmd = [sys.executable, os.path.abspath(__file__), "--system", system, "--batch", str(batch),
           "--workload", workload, "--num-requests", str(n)]
    if overrides:
        cmd += ["--overrides", json.dumps(overrides)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        print(out.stdout[-3000:], out.stderr[-3000:])
        raise RuntimeError(f"{system} bs={batch} failed")
    return json.loads(line[7:])


def sweep(args):
    results = []
    for workload in ("mixed", "fixed"):
        for batch in args.batches:
            hf = sub("hf", batch, workload, args.num_requests)
            mi = sub("minfer", batch, workload, args.num_requests)
            results += [hf, mi]
            print(f"{workload:5s} bs={batch:3d}  HF {hf['output_tok_per_s']:8.1f} tok/s ({hf['seconds']:6.1f}s)   "
                  f"minfer {mi['output_tok_per_s']:8.1f} tok/s ({mi['seconds']:6.1f}s)   "
                  f"speedup {mi['output_tok_per_s'] / hf['output_tok_per_s']:.2f}x", flush=True)
    # minfer with no concurrency cap (scheduler limited only by KV-cache capacity)
    for workload in ("mixed", "fixed"):
        mi = sub("minfer", 256, workload, args.num_requests)
        results.append(mi)
        print(f"{workload:5s} minfer uncapped (max_num_seqs=256): {mi['output_tok_per_s']:.1f} tok/s", flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "throughput.json"), "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "model": MODEL, "results": results}, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["hf", "minfer"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workload", choices=["mixed", "fixed"], default="mixed")
    ap.add_argument("--num-requests", type=int, default=256)
    ap.add_argument("--overrides", default=None, help="JSON EngineConfig overrides for minfer")
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 64])
    args = ap.parse_args()
    single(args) if args.system else sweep(args)
