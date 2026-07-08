"""Self-supervised ORACLE mode labels for the adaptive-prediction gate (M3).

No human annotation. Raw VLA data already pairs each state with a ground-truth
action chunk, so "was future prediction necessary here?" is answerable by the
frozen dual-regime WAM itself: run it once per mode {SKIP, LATENT, FULL} on the
same state and measure each mode's action error against the dataset chunk. The
oracle label is the CHEAPEST mode whose error is within tolerance of FULL's
("cheapest sufficient mode"):

    label = min{ i in MODE_ORDER : err(i) <= err(FULL) * (1 + tol_rel) + tol_abs }

FULL always satisfies its own bound, so a label always exists. Errors are
computed in the NORMALIZED action space (the dataset processor's output = the
space `infer_action` denoises in), masked by `action_is_pad`, and optionally
truncated to the executed prefix (`exec_horizon` = the eval `replan_steps`;
prediction quality past the replan point never affects control).

Shards store the per-step error CURVES, not just the chosen label, so labels can
be re-derived OFFLINE with different tolerances / metrics / horizons via
`relabel_from_steps` — no WAM re-run (the expensive part) needed.

The BC (SFT) consumer lives in RLinf (`rlinf/models/embodiment/gate_policy/bc.py`)
and reads shards written by `scripts/generate_gate_oracle_labels.py`.
"""
from __future__ import annotations

import glob as _glob
import os
from typing import Any, Optional, Sequence

import torch

from .modes import MODE_ORDER, NUM_MODES

LABEL_SHARD_VERSION = 1
FULL_INDEX = NUM_MODES - 1  # MODE_ORDER is (SKIP, LATENT, FULL); FULL is last.

# Per-sample tensors every shard must carry ([N, ...] stacked over samples).
SHARD_DATA_KEYS = (
    "world_feat",   # [N, z]    gate input, cheap current-context latent
    "proprio",      # [N, P]    gate input, normalized proprio at t0
    "label",        # [N]       oracle mode index (int64, MODE_ORDER order)
    "chunk_err",    # [N, 3]    per-mode chunk error used for the stored label
    "step_l1",      # [N, 3, T] per-step L1 error curves (offline relabeling)
    "step_l2",      # [N, 3, T] per-step L2 error curves (offline relabeling)
    "valid_steps",  # [N, T]    bool, ~action_is_pad
    "sample_idx",   # [N]       dataset index the record came from (int64)
)


def per_step_errors(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-step L1/L2 error between two action chunks [T, A] -> ([T], [T])."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    diff = pred.detach().float().cpu() - gt.detach().float().cpu()
    return diff.abs().mean(dim=-1), diff.pow(2).mean(dim=-1).sqrt()


@torch.no_grad()
def compute_mode_step_errors(
    adapter,
    *,
    input_image: torch.Tensor,
    gt_action: torch.Tensor,
    proprio: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    prompt: Optional[str] = None,
    seeds: Sequence[int] = (0,),
    world_feat: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Run the FROZEN WAM once per (mode, seed) on one state; return error curves.

    `gt_action` is the dataset's NORMALIZED action chunk [T, A] and defines the
    action horizon. Seeds are shared across modes (paired comparison: the same
    initial action noise, so error differences come from the conditioning, not
    the draw); >1 seed averages out inference stochasticity.

    Returns {world_feat [z], step_l1 [3, T], step_l2 [3, T], costs [3]}.
    """
    if gt_action.ndim != 2:
        raise ValueError(f"`gt_action` must be [T, A], got {tuple(gt_action.shape)}")
    if len(seeds) < 1:
        raise ValueError("`seeds` must contain at least one seed.")
    horizon = int(gt_action.shape[0])
    if world_feat is None:
        world_feat = adapter.encode_world_feat(input_image)

    step_l1 = torch.zeros(NUM_MODES, horizon)
    step_l2 = torch.zeros(NUM_MODES, horizon)
    costs = torch.zeros(NUM_MODES)
    for mode_idx, mode in enumerate(MODE_ORDER):
        for seed in seeds:
            out = adapter.act(
                input_image=input_image,
                mode=mode,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                prompt=prompt,
                action_horizon=horizon,
                world_feat=world_feat,
                seed=int(seed),
            )
            l1, l2 = per_step_errors(out["action_chunk"], gt_action)
            step_l1[mode_idx] += l1
            step_l2[mode_idx] += l2
            costs[mode_idx] = float(out["cost"])
        step_l1[mode_idx] /= len(seeds)
        step_l2[mode_idx] /= len(seeds)

    return {
        "world_feat": world_feat.detach().float().cpu(),
        "step_l1": step_l1,
        "step_l2": step_l2,
        "costs": costs,
    }


def chunk_errors_from_steps(
    step_err: torch.Tensor,
    valid_steps: torch.Tensor,
    *,
    exec_horizon: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-step error curves to per-mode chunk errors.

    step_err [..., 3, T], valid_steps [..., T] bool ->
    (chunk_err [..., 3], has_valid [...] bool). Mean over valid steps within the
    first `exec_horizon` steps; rows with no valid step get chunk_err=+inf and
    has_valid=False (callers should drop or FULL-label them).
    """
    if step_err.shape[-2] != NUM_MODES:
        raise ValueError(f"`step_err` second-to-last dim must be {NUM_MODES}, got {tuple(step_err.shape)}")
    if step_err.shape[-1] != valid_steps.shape[-1]:
        raise ValueError(
            f"step/valid horizon mismatch: {step_err.shape[-1]} vs {valid_steps.shape[-1]}"
        )
    valid = valid_steps.bool()
    if exec_horizon is not None:
        if int(exec_horizon) < 1:
            raise ValueError(f"`exec_horizon` must be >= 1, got {exec_horizon}")
        valid = valid.clone()
        valid[..., int(exec_horizon):] = False
    weights = valid.unsqueeze(-2).to(step_err.dtype)  # [..., 1, T]
    counts = weights.sum(dim=-1)  # [..., 1]
    chunk = (step_err.float() * weights).sum(dim=-1) / counts.clamp_min(1.0)
    has_valid = counts.squeeze(-1) > 0
    chunk = torch.where(
        has_valid.unsqueeze(-1).expand_as(chunk), chunk, torch.full_like(chunk, float("inf"))
    )
    return chunk, has_valid


def select_cheapest_sufficient(
    chunk_err: torch.Tensor,
    *,
    tol_abs: float = 0.0,
    tol_rel: float = 0.0,
) -> torch.Tensor:
    """Cheapest-sufficient oracle label from per-mode chunk errors [..., 3] -> [...].

    Sufficient: err(mode) <= err(FULL) * (1 + tol_rel) + tol_abs. Modes are tried
    in MODE_ORDER (cheapest first); FULL always qualifies, so rows where nothing
    compares True (e.g. all-inf from empty masks) also fall back to FULL.
    """
    if float(tol_abs) < 0.0 or float(tol_rel) < 0.0:
        raise ValueError(f"tolerances must be >= 0, got tol_abs={tol_abs}, tol_rel={tol_rel}")
    if chunk_err.shape[-1] != NUM_MODES:
        raise ValueError(f"`chunk_err` last dim must be {NUM_MODES}, got {tuple(chunk_err.shape)}")
    err = chunk_err.float()
    threshold = err[..., FULL_INDEX] * (1.0 + float(tol_rel)) + float(tol_abs)
    # non-finite errors (inf from empty masks, NaN) are never "sufficient";
    # without the guard `inf <= inf` would wrongly qualify SKIP on all-inf rows.
    sufficient = (err <= threshold.unsqueeze(-1)) & torch.isfinite(err)  # [..., 3]
    # argmax on int returns the FIRST max index == first sufficient (cheapest).
    labels = sufficient.long().argmax(dim=-1)
    labels = torch.where(
        sufficient.any(dim=-1), labels, torch.full_like(labels, FULL_INDEX)
    )
    return labels


def relabel_from_steps(
    step_l1: torch.Tensor,
    step_l2: torch.Tensor,
    valid_steps: torch.Tensor,
    *,
    metric: str = "l1",
    exec_horizon: Optional[int] = None,
    tol_abs: float = 0.0,
    tol_rel: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """OFFLINE relabeling from stored error curves (no WAM re-run).

    Inputs are the shard tensors ([N, 3, T], [N, 3, T], [N, T]). Returns
    (labels [N], chunk_err [N, 3], has_valid [N]).
    """
    if metric not in ("l1", "l2"):
        raise ValueError(f"`metric` must be 'l1' or 'l2', got {metric!r}")
    step_err = step_l1 if metric == "l1" else step_l2
    chunk_err, has_valid = chunk_errors_from_steps(
        step_err, valid_steps, exec_horizon=exec_horizon
    )
    labels = select_cheapest_sufficient(chunk_err, tol_abs=tol_abs, tol_rel=tol_rel)
    return labels, chunk_err, has_valid


def label_distribution(labels: torch.Tensor) -> dict[str, float]:
    """Fraction of each mode in `labels` (the first sanity metric to inspect)."""
    total = max(int(labels.numel()), 1)
    return {
        mode.value: float((labels == idx).sum().item()) / total
        for idx, mode in enumerate(MODE_ORDER)
    }


# ----- shard IO ----------------------------------------------------------- #
def write_label_shard(path: str, *, data: dict[str, torch.Tensor], meta: dict[str, Any]) -> None:
    """Write one oracle-label shard (torch.save payload).

    `data` must carry every key in SHARD_DATA_KEYS with a shared leading N.
    `meta` should record everything needed to interpret/reproduce the shard
    (task, ckpt, backbone_kind, k_lo/k_hi, seeds, tolerances, cost table, ...);
    `mode_order` is stamped in automatically and validated on load.
    """
    missing = [k for k in SHARD_DATA_KEYS if k not in data]
    if missing:
        raise ValueError(f"label shard is missing data keys: {missing}")
    sizes = {k: int(data[k].shape[0]) for k in SHARD_DATA_KEYS}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"inconsistent leading dims across shard tensors: {sizes}")
    payload = {
        "version": LABEL_SHARD_VERSION,
        "meta": {**dict(meta), "mode_order": [m.value for m in MODE_ORDER]},
        "data": {k: data[k].detach().cpu() for k in SHARD_DATA_KEYS},
    }
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    torch.save(payload, path)


def resolve_shard_paths(shards: str | Sequence[str]) -> list[str]:
    """Accept a glob pattern, a single path, or a list of either; return sorted paths."""
    patterns = [shards] if isinstance(shards, str) else list(shards)
    paths: list[str] = []
    for pattern in patterns:
        matched = sorted(_glob.glob(os.path.expanduser(str(pattern))))
        if matched:
            paths.extend(matched)
        elif os.path.isfile(os.path.expanduser(str(pattern))):
            paths.append(os.path.expanduser(str(pattern)))
    # de-dupe, keep order
    seen: set[str] = set()
    unique = [p for p in paths if not (p in seen or seen.add(p))]
    if not unique:
        raise FileNotFoundError(f"no oracle-label shards matched: {shards!r}")
    return unique


def load_label_shards(shards: str | Sequence[str]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load + concatenate shards; validate they were produced compatibly.

    Returns (data, meta): `data` has SHARD_DATA_KEYS stacked over all shards;
    `meta` is the first shard's meta plus {num_shards, shard_paths, num_samples}.
    Shards must agree on mode_order and on every trailing tensor shape.
    """
    paths = resolve_shard_paths(shards)
    datas: list[dict[str, torch.Tensor]] = []
    metas: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("version", -1)) != LABEL_SHARD_VERSION:
            raise ValueError(
                f"{path}: unsupported shard version {payload.get('version')!r} "
                f"(expected {LABEL_SHARD_VERSION})"
            )
        meta = payload["meta"]
        if list(meta.get("mode_order", [])) != [m.value for m in MODE_ORDER]:
            raise ValueError(
                f"{path}: shard mode_order {meta.get('mode_order')!r} does not match "
                f"the current MODE_ORDER {[m.value for m in MODE_ORDER]}"
            )
        missing = [k for k in SHARD_DATA_KEYS if k not in payload["data"]]
        if missing:
            raise ValueError(f"{path}: shard is missing data keys: {missing}")
        datas.append(payload["data"])
        metas.append(meta)
    reference = {k: tuple(datas[0][k].shape[1:]) for k in SHARD_DATA_KEYS}
    for path, data in zip(paths, datas):
        shapes = {k: tuple(data[k].shape[1:]) for k in SHARD_DATA_KEYS}
        if shapes != reference:
            raise ValueError(
                f"{path}: shard tensor shapes {shapes} do not match the first shard {reference}"
            )
    merged = {k: torch.cat([d[k] for d in datas], dim=0) for k in SHARD_DATA_KEYS}
    meta = dict(metas[0])
    meta.update(
        num_shards=len(paths),
        shard_paths=paths,
        num_samples=int(merged["label"].shape[0]),
    )
    return merged, meta
