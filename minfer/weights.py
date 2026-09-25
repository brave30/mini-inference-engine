import glob
import os

import torch
from safetensors.torch import load_file


def resolve_model_path(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download
    return snapshot_download(name_or_path, allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    state = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        state.update(load_file(f, device="cpu"))
    if not state:
        raise FileNotFoundError(f"no .safetensors files in {path}")
    return state
