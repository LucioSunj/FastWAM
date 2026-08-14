"""Strict helpers for deterministic, offline FastWAM text embeddings."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import torch


def encoder_id_from_model_id(model_id: str) -> str:
    """Return the filename-safe encoder id used by FastWAM caches."""

    base = str(model_id).split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "", base.lower()) or "textenc"


def prompt_sha256(prompt: str) -> str:
    """Hash the exact prompt string used to address the offline cache."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def prompt_cache_path(
    cache_dir: str | Path,
    prompt: str,
    *,
    context_len: int = 128,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
) -> Path:
    """Resolve the canonical cache filename for one exact prompt."""

    encoder_id = encoder_id_from_model_id(model_id)
    filename = f"{prompt_sha256(prompt)}.t5_len{int(context_len)}.{encoder_id}.pt"
    return Path(cache_dir).expanduser().resolve() / filename


def load_text_embedding(
    cache_dir: str | Path,
    prompt: str,
    *,
    context_len: int = 128,
    text_dim: int = 4096,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, Path]:
    """Load one cached T5 context and reproduce online padding semantics."""

    path = prompt_cache_path(
        cache_dir,
        prompt,
        context_len=context_len,
        model_id=model_id,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing required encoded prompt cache: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"context", "mask"}:
        raise ValueError(f"Invalid encoded prompt payload: {path}")
    context, mask = payload["context"], payload["mask"]
    if not torch.is_tensor(context) or not torch.is_tensor(mask):
        raise TypeError(f"Encoded prompt context/mask must be tensors: {path}")
    if tuple(context.shape) != (int(context_len), int(text_dim)):
        raise ValueError(
            f"Encoded prompt context shape changed: {tuple(context.shape)} at {path}"
        )
    if tuple(mask.shape) != (int(context_len),):
        raise ValueError(f"Encoded prompt mask shape changed: {tuple(mask.shape)}")
    if not bool(torch.isfinite(context.float()).all().item()):
        raise ValueError(f"Encoded prompt contains non-finite values: {path}")

    context = context.to(dtype=dtype).contiguous()
    mask = mask.to(dtype=torch.bool).contiguous()
    context = context.clone()
    context[~mask] = 0
    mask = torch.ones_like(mask)
    return (
        context.unsqueeze(0).to(device=device, non_blocking=True),
        mask.unsqueeze(0).to(device=device, non_blocking=True),
        path,
    )
