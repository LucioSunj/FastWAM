"""Read the text cache formats used by FastWAM and EasyWAM parents."""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch


def validate_text_padding(text_padding: str) -> str:
    """Validate the text attention semantics selected for a parent model."""
    if text_padding not in {"legacy_visible", "masked"}:
        raise ValueError(
            f"text_padding must be 'legacy_visible' or 'masked', got {text_padding!r}."
        )
    return text_padding


def validate_checkpoint_text_padding(payload: Mapping, expected: str) -> None:
    """Reject weights trained with different text attention semantics."""
    validate_text_padding(expected)
    actual = payload.get("text_padding", "legacy_visible")
    validate_text_padding(actual)
    if actual != expected:
        raise ValueError(
            f"Checkpoint text_padding={actual!r}; configured model expects {expected!r}."
        )


def load_text_context(
    cache_dir: str | Path,
    prompt: str,
    context_len: int,
    text_dim: int | None = None,
    text_padding: str = "legacy_visible",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a CPU text context and its attention mask.

    Args:
        cache_dir: Directory containing the selected parent's text caches.
        prompt: Complete prompt, including the dataset instruction prefix.
        context_len: Expected number of cached text tokens.
        text_dim: Expected embedding width, when known by the caller.
        text_padding: Legacy visible padding or EasyWAM valid-token masking.

    Returns:
        A context tensor shaped [L, D] and a boolean mask shaped [L].
        Legacy cache dtypes are preserved; EasyWAM caches use BF16.
    """
    validate_text_padding(text_padding)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    family = "text" if text_padding == "masked" else "t5"
    encoder_id = "wan22ti2v5b"
    path = Path(cache_dir) / f"{digest}.{family}_len{context_len}.{encoder_id}.pt"
    if not path.is_file():
        project = "EasyWAM" if text_padding == "masked" else "FastWAM"
        raise FileNotFoundError(
            f"Missing text embedding cache: {path}. "
            f"Run {project}/scripts/precompute_text_embeds.py for this task."
        )
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    context = payload["context"]
    mask = payload["mask"]
    if context.ndim != 2 or context.shape[0] != context_len:
        raise ValueError(f"Cached context must have shape [{context_len}, D]: {path}.")
    if mask.ndim != 1 or mask.shape[0] != context_len:
        raise ValueError(f"Cached mask must have shape [{context_len}]: {path}.")
    if text_dim is not None and context.shape[1] != text_dim:
        raise ValueError(f"Cached context width must be {text_dim}: {path}.")
    if text_padding == "masked":
        expected_metadata = {
            "format_version": 3,
            "encoder_id": encoder_id,
            "context_len": context_len,
            "prompt_hash": digest,
        }
        for key, expected in expected_metadata.items():
            if key in payload and payload[key] != expected:
                raise ValueError(
                    f"EasyWAM text cache {key}={payload[key]!r}; expected {expected!r}."
                )
        if context.dtype != torch.bfloat16 or mask.dtype != torch.bool:
            raise TypeError(
                "EasyWAM text caches require BF16 context and boolean mask."
            )
        return context, mask
    context = context.clone()
    mask = mask.bool()
    context[~mask] = 0.0
    return context, torch.ones_like(mask)
