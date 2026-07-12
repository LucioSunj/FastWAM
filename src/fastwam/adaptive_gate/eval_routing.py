"""Explicit branch selection for evaluation entrypoints."""
from __future__ import annotations

import inspect


def explicit_eval_branch(
    model,
    method_name: str,
    requested_branch: str = "base",
    *,
    require_video: bool = False,
) -> dict[str, str]:
    """Return ``force_branch`` kwargs for adaptive models, otherwise ``{}``."""
    method = getattr(model, method_name)
    if "force_branch" not in inspect.signature(method).parameters:
        return {}
    future_branch = getattr(model, "adaptive_backbone_kind", None)
    if future_branch not in {"idm", "joint"}:
        raise ValueError(
            "Routing-capable model must declare adaptive_backbone_kind='idm' or 'joint'."
        )
    branch = str(requested_branch)
    valid = {"base", str(future_branch)}
    if branch not in valid:
        raise ValueError(f"force_branch must be one of {sorted(valid)}, got {branch!r}.")
    if require_video and branch != future_branch:
        raise ValueError(
            f"Video visualization requires force_branch={future_branch!r}, got {branch!r}."
        )
    return {"force_branch": branch}
