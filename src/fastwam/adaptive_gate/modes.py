"""Mode definitions for the adaptive-prediction gate.

Three prediction modes the gate chooses among, in increasing compute:
- SKIP   : no future-video prediction (reactive); action conditions on the
           current-context latent only. Maps to the frozen model's `base` branch
           (FastWAM.infer_action; first-frame conditioning, no video denoising).
- LATENT : run the video branch for a FEW denoising steps (`k_lo`); action
           conditions on the intermediate self-generated future latent (no pixel
           decode). Maps to the `joint`/`idm` branch with `num_inference_steps=k_lo`.
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
    return MODE_ORDER[int(index)]


def mode_to_index(mode) -> int:
    return MODE_ORDER.index(WAMMode(mode))


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
    """Map a mode to (force_branch, num_inference_steps) for `model.infer_action`.

    - SKIP  -> ("base",  action_steps)  # video not denoised; this drives the
                                          # action denoising loop (current behavior).
    - LATENT-> (future,  k_lo)
    - FULL  -> (future,  k_hi)

    NOTE (default scheme, see decision #3): for LATENT/FULL the single
    `num_inference_steps` drives BOTH the video and action denoising loops, which
    is the frozen model's native coupling (fastwam.py:866 zip / idm two-stage).
    # TODO(step-decouple): a cleaner LATENT would run the VIDEO branch for k_lo
    #   steps but still refine the ACTION for the full schedule. That needs a
    #   small refactor of FastWAMJoint.infer_action / FastWAMIDM.infer_joint to
    #   accept separate video/action step counts. Left as a follow-up.
    """
    mode = WAMMode(mode)
    future = future_branch_for(backbone_kind)
    if mode is WAMMode.SKIP:
        return "base", int(action_steps)
    if mode is WAMMode.LATENT:
        return future, int(k_lo)
    return future, int(k_hi)  # FULL
