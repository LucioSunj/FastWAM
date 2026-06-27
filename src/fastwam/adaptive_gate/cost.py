"""Per-mode compute cost for the adaptive-prediction gate.

The reward has a per-step compute penalty `-lambda * cost(mode)`, with
`cost(FULL) = 1` by definition. `cost(mode)` is the RELATIVE compute of running
the world-action model in that mode.

DESIGN (what "cost" means; decision #6)
---------------------------------------
"Compute" is best measured as FLOPs, because it is hardware-independent and is
exactly the quantity the gate is meant to economize. The dominant *variable*
cost across modes is the video-branch denoising loop: SKIP runs 0 video-denoise
steps, LATENT runs `k_lo`, FULL runs `k_hi`. All modes share a roughly-fixed
overhead (the single VAE encode + the action-denoising loop). So:

    total(mode) ~= C_fixed + C_video_step * video_steps(mode)
    cost(mode)   = total(mode) / total(FULL)            # normalized, cost(FULL)=1

We provide cost in two ways:
  1. MEASURED (preferred): `scripts/profile_wam_modes.py` runs each mode once with
     a FLOP counter (and also times wall-clock), normalizes by FULL, and writes a
     cost YAML. The reward then reads `source: flops` (or `latency`) from it.
  2. ANALYTICAL FALLBACK: when no measured table exists, `default_cost_table`
     approximates the formula above with a single `fixed_fraction` knob
     (C_fixed / total(FULL)). This keeps everything runnable out-of-the-box; the
     numbers are placeholders until profiling is run.

Why not just wall-clock latency? Latency is hardware/implementation specific and
noisier; we record it for reference but default the *reward* to FLOPs. Why not a
flat per-step constant? Because LATENT genuinely costs less than FULL, and the
gate should be able to prefer it — a flat constant would erase that signal.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from .modes import MODE_ORDER, WAMMode


def default_cost_table(
    k_lo: int,
    k_hi: int,
    *,
    fixed_fraction: float = 0.15,
) -> dict[str, float]:
    """Analytical fallback cost table (normalized so cost(FULL)=1).

    cost(mode) = fixed_fraction + (1 - fixed_fraction) * video_steps(mode) / k_hi
    with video_steps = {SKIP: 0, LATENT: k_lo, FULL: k_hi}.

    `fixed_fraction` ~= the share of FULL's compute that is NOT the video loop
    (VAE encode + action denoising). It is a rough placeholder; run the profiler
    to replace these with measured FLOPs.

    # TODO(measure): replace the analytical default with measured FLOPs from
    #   scripts/profile_wam_modes.py for the specific backbone / resolution / K.
    """
    k_lo = int(k_lo)
    k_hi = int(k_hi)
    if k_hi <= 0:
        raise ValueError(f"k_hi must be positive, got {k_hi}.")
    if not 0.0 <= float(fixed_fraction) < 1.0:
        raise ValueError(f"fixed_fraction must be in [0, 1), got {fixed_fraction}.")
    var = 1.0 - float(fixed_fraction)
    video_steps = {WAMMode.SKIP: 0, WAMMode.LATENT: k_lo, WAMMode.FULL: k_hi}
    table = {
        m.value: float(fixed_fraction) + var * (video_steps[m] / k_hi)
        for m in MODE_ORDER
    }
    # cost(FULL) is exactly 1.0 by construction.
    table[WAMMode.FULL.value] = 1.0
    return table


def normalize_cost_table(raw: dict[str, float]) -> dict[str, float]:
    """Normalize a raw per-mode measure (FLOPs or latency) so cost(FULL)=1."""
    full = float(raw[WAMMode.FULL.value])
    if full <= 0:
        raise ValueError("FULL-mode raw cost must be positive to normalize.")
    return {m.value: float(raw[m.value]) / full for m in MODE_ORDER}


def save_cost_table(
    path: str,
    *,
    normalized: dict[str, float],
    source: str = "flops",
    raw: Optional[dict[str, dict[str, float]]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    """Write a cost YAML consumed by the reward / adapter."""
    import yaml  # local import: only needed when actually profiling/writing

    payload: dict[str, Any] = {
        "source": str(source),
        "modes": {k: float(v) for k, v in normalized.items()},
    }
    if raw is not None:
        payload["raw"] = raw
    if meta is not None:
        payload["meta"] = meta
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_cost_table(path: Optional[str], *, source: Optional[str] = None) -> Optional[dict[str, float]]:
    """Load a normalized per-mode cost table from a YAML written by the profiler.

    Returns None if `path` is None or missing (caller should fall back to
    `default_cost_table`). `source` overrides which raw measure to normalize
    ("flops" | "latency"); if None, uses the file's `modes` block as-is.
    """
    if not path or not os.path.isfile(path):
        return None
    import yaml

    with open(path) as handle:
        payload = yaml.safe_load(handle)
    if source is not None and "raw" in payload and source in payload["raw"]:
        return normalize_cost_table(payload["raw"][source])
    modes = payload["modes"]
    return {m.value: float(modes[m.value]) for m in MODE_ORDER}
