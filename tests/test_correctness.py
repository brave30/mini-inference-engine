"""End-to-end check: minfer greedy decoding reproduces Hugging Face greedy decoding.

Runs minfer under several configs (Triton kernels + CUDA graphs, tiny chunk budget that
forces multi-chunk prefill, pure-PyTorch paths, and a KV cache small enough to force
preemption) and compares generated token ids with HF `generate(do_sample=False)`.

Different (equally valid) bf16 reduction orders can flip an argmax between two logits
that are tied to within bf16 precision, after which greedy continuations legitimately
differ. So for every sequence that diverges, we run HF on the shared prefix and require
that the token minfer picked is within 2 bf16 ULPs of HF's top logit (a tie), i.e. that
no divergence is caused by a real bug.
Run: python tests/test_correctness.py
"""
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minfer import LLMEngine, SamplingParams
from minfer.config import EngineConfig

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW = 64
PROMPTS = [
    "The capital of France is",
    "Explain how a paged KV cache works in an LLM inference server.",
    "def fibonacci(n):",
    "Once upon a time, in a small village near the mountains,",
    "List three differences between TCP and UDP:",
    "The quick brown fox",
    "Translate to Spanish: I would like a cup of coffee, please.",
    "In 1969, Apollo 11",
    "Write a haiku about GPUs.",
    "Q: What is 17 * 23?\nA:",
    "Summarize the plot of Romeo and Juliet in two sentences, making sure to mention how the story "
    "ends and why the two families were feuding in the first place. " * 3,
    "Hello",
]
CONFIGS = {
    "triton + cuda graphs (default)": EngineConfig(),
    "chunked prefill, 48-token budget": EngineConfig(max_num_batched_tokens=48),
    "pytorch eager (no triton/graphs)": EngineConfig(use_triton_rmsnorm=False, use_triton_decode_attn=False,
                                                     use_cuda_graphs=False),
    "tiny kv cache (forces preemption)": EngineConfig(kv_cache_memory_fraction=0.0035, use_cuda_graphs=False),
}


def bf16_ulp(x: float) -> float:
    return 2.0 ** (torch.tensor(abs(x)).log2().floor().item() - 7)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = [tok(p).input_ids for p in PROMPTS]
    sp = SamplingParams(max_tokens=MAX_NEW)  # both sides stop at EOS

    ours = {}
    for name, cfg in CONFIGS.items():
        eng = LLMEngine(MODEL, cfg, verbose=False)
        ours[name] = [o.output_ids for o in eng.generate(prompt_ids, sp, decode_text=False)]
        if "preemption" in name:
            assert eng.scheduler.num_preemptions > 0, "preemption path not exercised"
            print(f"(preemption config triggered {eng.scheduler.num_preemptions} preemptions)")
        del eng
        gc.collect()
        torch.cuda.empty_cache()

    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    ref = []
    for ids in prompt_ids:  # unbatched: no padding effects in the reference
        g = hf.generate(torch.tensor([ids], device="cuda"), max_new_tokens=MAX_NEW, do_sample=False,
                        pad_token_id=tok.eos_token_id)
        ref.append(g[0, len(ids):].tolist())

    ok = True
    for name, got in ours.items():
        exact, matched, total, gaps = 0, 0, 0, []
        for ids, r, g in zip(prompt_ids, ref, got):
            total += len(r)
            d = next((i for i, (a, b) in enumerate(zip(r, g)) if a != b), None)
            if d is None and len(r) == len(g):
                exact += 1
                matched += len(r)
                continue
            d = d if d is not None else min(len(r), len(g))
            matched += d
            with torch.no_grad():
                logits = hf(torch.tensor([ids + r[:d]], device="cuda")).logits[0, -1].float()
            top = logits.max().item()
            gap = top - logits[g[d]].item() if d < len(g) else top - logits[tok.eos_token_id].item()
            gaps.append(gap)
            if gap > 2 * bf16_ulp(top):
                ok = False
                print(f"  REAL MISMATCH in {name!r}: prompt {PROMPTS[prompt_ids.index(ids)][:40]!r} "
                      f"step {d}, logit gap {gap:.3f}")
        print(f"{name:34s} identical sequences {exact}/{len(ref)}, tokens before first divergence "
              f"{matched}/{total} ({100 * matched / total:.1f}%), divergence logit gaps {[round(x, 4) for x in gaps]}")
    assert ok, "a divergence was not explained by a bf16 tie"
    print("correctness ok: every divergence from HF is a bf16-precision tie")


if __name__ == "__main__":
    main()
