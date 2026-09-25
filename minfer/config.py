import json
import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    eos_token_id: int

    @classmethod
    def from_pretrained(cls, path: str) -> "ModelConfig":
        with open(os.path.join(path, "config.json")) as f:
            c = json.load(f)
        eos = c.get("eos_token_id", 2)
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_layers=c["num_hidden_layers"],
            num_heads=c["num_attention_heads"],
            num_kv_heads=c.get("num_key_value_heads", c["num_attention_heads"]),
            head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=c.get("rope_theta", 10000.0),
            max_position_embeddings=c["max_position_embeddings"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
            # Qwen2 uses a bias on q/k/v but does not record it in attention_bias.
            attention_bias=c.get("attention_bias", False) or c.get("model_type") == "qwen2",
            eos_token_id=eos[0] if isinstance(eos, list) else eos,
        )


@dataclass
class EngineConfig:
    block_size: int = 16                 # tokens per KV-cache page
    max_num_seqs: int = 256              # max concurrently running sequences
    max_num_batched_tokens: int = 2048   # per-step token budget (chunked-prefill chunk cap)
    kv_cache_memory_fraction: float = 0.85  # of free GPU memory after weights load
    max_model_len: int = 2048
    enable_chunked_prefill: bool = True
    use_triton_rmsnorm: bool = True
    use_triton_decode_attn: bool = True
    use_cuda_graphs: bool = True
    cuda_graph_batch_sizes: tuple = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256)
