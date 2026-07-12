"""Canonical modes for the adaptive world-model gate.

The gate deliberately exposes only two choices:

``UNCOND``
    The reactive/base FastWAM action path.  It conditions on the encoded current
    frame and performs no future-video denoising.

``IDM``
    The complete IDM path.  It generates the full future latent with the frozen
    world model and then conditions the action expert on that latent.

Keeping this surface binary is intentional: a low-NFE solver is not an
"intermediate latent" from the full trajectory, and mixing solver quality with
the routing decision makes the learned gate difficult to interpret.
"""
from __future__ import annotations

import enum
import numbers


class WAMMode(str, enum.Enum):
    UNCOND = "uncond"
    IDM = "idm"


# Stable categorical contract shared by FastWAM, RLinf, cost files and labels.
MODE_ORDER: tuple[WAMMode, ...] = (WAMMode.UNCOND, WAMMode.IDM)
NUM_MODES = len(MODE_ORDER)


def mode_from_index(index: int) -> WAMMode:
    if isinstance(index, bool) or not isinstance(index, numbers.Integral):
        raise ValueError(f"mode index must be an integer, got {index!r}.")
    idx = int(index)
    if idx < 0 or idx >= NUM_MODES:
        raise ValueError(f"mode index {idx} out of range [0, {NUM_MODES}).")
    return MODE_ORDER[idx]


def coerce_mode(mode) -> WAMMode:
    """Accept canonical string/enum values or categorical indices."""
    if isinstance(mode, WAMMode):
        return mode
    if isinstance(mode, str):
        if mode in {"0", "1"}:
            return mode_from_index(int(mode))
        try:
            return WAMMode(mode)
        except ValueError:
            pass
    elif isinstance(mode, numbers.Integral) and not isinstance(mode, bool):
        return mode_from_index(mode)
    raise ValueError(
        f"Unknown WAM mode `{mode}`. Expected one of "
        f"{[m.value for m in MODE_ORDER]} or an integer index in [0, {NUM_MODES})."
    )


def mode_to_index(mode) -> int:
    return MODE_ORDER.index(coerce_mode(mode))


def mode_to_branch_steps(mode, *, inference_steps: int) -> tuple[str, int]:
    """Map a categorical mode to the frozen model branch and solver depth."""
    steps = int(inference_steps)
    if steps <= 0:
        raise ValueError(f"inference_steps must be positive, got {inference_steps}.")
    selected = coerce_mode(mode)
    return ("base", steps) if selected is WAMMode.UNCOND else ("idm", steps)
