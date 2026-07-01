"""Mode definitions for the adaptive-prediction gate.

Three prediction modes the gate chooses among, in increasing compute:
- SKIP   : no future-video prediction (reactive); action conditions on the
           current-context latent only. Maps to the frozen model's `base` branch
           (FastWAM.infer_action; first-frame conditioning, no video denoising).
- LATENT : run the video branch for a FEW denoising steps (`k_lo`); action
           conditions on the intermediate self-generated future latent (no pixel
           decode). IDM can keep full action refinement; joint keeps its native
           coupled video/action schedule.
- FULL   : run the video branch for the FULL schedule (`k_hi`); action conditions
           on the refined self-generated future latent. Same branch, `k_hi` steps.

The gate is a 3-way categorical; `MODE_ORDER` fixes the index<->mode mapping so the
policy logits, the cost table, and logging all agree.
"""
from __future__ import annotations

import enum


class WAMMode(str, enum.Enum):
    SKIP = "skip"
    LATENT = "latent"
    FULL = "full"


# Canonical categorical order: logit index i  <->  MODE_ORDER[i].
# SKIP=0, LATENT=1, FULL=2. Do not reorder (configs/logs/labels depend on it).
MODE_ORDER: tuple[WAMMode, ...] = (WAMMode.SKIP, WAMMode.LATENT, WAMMode.FULL)
NUM_MODES = len(MODE_ORDER)

_VALID_BACKBONES = {"joint", "idm"}


def mode_from_index(index: int) -> WAMMode:
    idx = int(index)
    if idx < 0 or idx >= NUM_MODES:
        raise ValueError(f"mode index {idx} out of range [0, {NUM_MODES}).")
    return MODE_ORDER[idx]


def coerce_mode(mode) -> WAMMode:
    """Accept canonical mode values or categorical indices."""
    if isinstance(mode, WAMMode):
        return mode
    try:
        return WAMMode(mode)
    except ValueError:
        try:
            return mode_from_index(int(mode))
        except (TypeError, ValueError) as index_exc:
            raise ValueError(
                f"Unknown WAM mode `{mode}`. Expected one of {[m.value for m in MODE_ORDER]} "
                f"or an index in [0, {NUM_MODES})."
            ) from index_exc


def mode_to_index(mode) -> int:
    return MODE_ORDER.index(coerce_mode(mode))


def future_branch_for(backbone_kind: str) -> str:
    """The frozen model's branch name used by the LATENT/FULL modes.

    backbone_kind="joint" -> MetricAdaptiveFastWAMJoint's "joint" branch.
    backbone_kind="idm"   -> MetricAdaptiveFastWAM's "idm" branch.
    SKIP always uses the "base" branch on either model.
    """
    backbone_kind = str(backbone_kind)
    if backbone_kind not in _VALID_BACKBONES:
        raise ValueError(
            f"Unknown backbone_kind `{backbone_kind}`. Expected one of {sorted(_VALID_BACKBONES)}."
        )
    return "joint" if backbone_kind == "joint" else "idm"


def mode_to_branch_steps(
    mode,
    *,
    backbone_kind: str,
    k_lo: int,
    k_hi: int,
    action_steps: int,
) -> tuple[str, int]:
    """Backward-compatible mode mapping to a single inference-step count.

    Prefer `mode_to_branch_step_counts` for new adaptive-gate code.
    """
    branch, video_steps, action_step_count = mode_to_branch_step_counts(
        mode,
        backbone_kind=backbone_kind,
        k_lo=k_lo,
        k_hi=k_hi,
        action_steps=action_steps,
    )
    return branch, int(action_step_count if video_steps is None else video_steps)


def mode_to_branch_step_counts(
    mode,
    *,
    backbone_kind: str,
    k_lo: int,
    k_hi: int,
    action_steps: int,
) -> tuple[str, int | None, int]:
    """Map a mode to (force_branch, video_steps, action_steps).

    - SKIP   -> ("base", None, action_steps)
    - LATENT -> (future, k_lo, action_steps) for IDM
                (future, k_lo, k_lo) for joint, whose native inference is coupled
    - FULL   -> (future, k_hi, action_steps)

    `video_steps=None` means the selected branch does not run future-video
    denoising. For IDM, `action_steps` defaults to k_hi so LATENT changes only
    future-prediction compute while keeping action refinement full length. Joint
    inference denoises video/action together, so LATENT remains coupled.
    """
    mode = coerce_mode(mode)
    future = future_branch_for(backbone_kind)
    if mode is WAMMode.SKIP:
        return "base", None, int(action_steps)
    if mode is WAMMode.LATENT:
        if str(backbone_kind) == "joint":
            return future, int(k_lo), int(k_lo)
        return future, int(k_lo), int(action_steps)
    return future, int(k_hi), int(action_steps)  # FULL
