"""Lightweight inference-contract validation."""
from __future__ import annotations


ACTION_ONLY_ATTENTION_MODES = frozenset({"first_frame_causal", "per_frame_causal"})


def validate_action_only_attention_mode(mode: str) -> str:
    value = str(mode)
    if value not in ACTION_ONLY_ATTENTION_MODES:
        raise ValueError(
            "action-only inference requires video_attention_mask_mode in "
            f"{sorted(ACTION_ONLY_ATTENTION_MODES)}, got {value!r}."
        )
    return value
