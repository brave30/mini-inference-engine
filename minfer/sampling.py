from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0        # 0 = greedy
    ignore_eos: bool = False        # keep generating past EOS (for fixed-length benchmarks)


def sample(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """logits [B, V]; temperatures [B] (0 = greedy). Returns token ids [B]."""
    greedy = logits.argmax(dim=-1)
    if not bool((temperatures > 0).any()):
        return greedy
    t = torch.where(temperatures > 0, temperatures, 1.0)[:, None]
    probs = torch.softmax(logits.float() / t, dim=-1)
    # exponential-race trick: argmax(p / E) with E ~ Exp(1) is a sample from p, no CPU sync
    sampled = probs.div_(torch.empty_like(probs).exponential_()).argmax(dim=-1)
    return torch.where(temperatures > 0, sampled, greedy)
