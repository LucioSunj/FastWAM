"""Offline oracle analysis for binary UNCOND-versus-IDM routing."""
from __future__ import annotations

import glob as _glob
import hashlib
import json
import os
from typing import Any, Mapping, Optional, Sequence

import torch

from .features import pool_text_context
from .modes import MODE_ORDER, NUM_MODES, WAMMode

LABEL_SHARD_VERSION = 3
IDM_INDEX = MODE_ORDER.index(WAMMode.IDM)

SHARD_DATA_KEYS = (
    "world_feat",    # [N, 5*z] coarse spatial latent + channel spread
    "text_feat",     # [N, 64] deterministic pooled instruction feature
    "proprio",       # [N, P]
    "label",         # [N], UNCOND=0 / IDM=1
    "chunk_err",     # [N, 2]
    "step_l1",       # [N, 2, T]
    "step_l2",       # [N, 2, T]
    "valid_steps",   # [N, T]
    "sample_idx",    # [N]
    "group_id",      # [N], episode id or -1 when unavailable
    "best_err",      # [N], best attainable mode error
    "idm_err",       # [N], absolute quality of the expensive reference
    "idm_regret",    # [N], idm_err - best_err
)

SHARD_COMPAT_META_KEYS = (
    "task",
    "backbone_kind",
    "ckpt_fingerprint",
    "ckpt_file_sha256",
    "dataset_stats_fingerprint",
    "num_video_frames",
    "inference_steps",
    "solver_fingerprint",
    "context_len",
    "model_dtype",
    "cost_table",
    "metric",
    "exec_horizon",
    "tol_rel",
    "tol_abs",
    "num_seeds",
    "seed_base",
    "stride",
    "skip_padded",
    "max_samples",
    "num_shards",
    "world_feat_layout",
    "world_feat_dim",
    "text_feat_dim",
    "text_feat_layout",
    "proprio_dim",
    "action_horizon",
    "group_id_layout",
)


def compose_group_id(dataset_index, episode_index) -> int:
    """Pack a dataset/episode pair into one stable non-negative int64 id."""
    dataset_value = int(torch.as_tensor(dataset_index).reshape(-1)[0].item())
    episode_value = int(torch.as_tensor(episode_index).reshape(-1)[0].item())
    if dataset_value < 0 or episode_value < 0:
        return -1
    if dataset_value >= 2**31 or episode_value >= 2**32:
        raise ValueError(
            f"dataset/episode index too large to pack: {dataset_value}, {episode_value}."
        )
    return (dataset_value << 32) | episode_value


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def shard_compatibility_fingerprint(meta: Mapping[str, Any]) -> str:
    """Hash every semantic field that must agree before shards are merged."""
    semantic = {key: _jsonable(meta.get(key)) for key in SHARD_COMPAT_META_KEYS}
    encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def per_step_errors(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    diff = pred.detach().float().cpu() - gt.detach().float().cpu()
    return diff.abs().mean(dim=-1), diff.pow(2).mean(dim=-1).sqrt()


def all_mode_errors_finite(*errors: torch.Tensor) -> bool:
    """True only when every stored mode/error entry is finite."""
    return bool(errors) and all(bool(torch.isfinite(error).all()) for error in errors)


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
    encoded_state=None,
) -> dict[str, Any]:
    """Run paired UNCOND/IDM inference and retain complete error curves."""
    if gt_action.ndim != 2:
        raise ValueError(f"`gt_action` must be [T,A], got {tuple(gt_action.shape)}")
    if not seeds:
        raise ValueError("`seeds` must contain at least one seed.")
    if context is None or context_mask is None:
        raise ValueError("Oracle generation requires context and context_mask for text_feat.")
    if encoded_state is None:
        encoded_state = adapter.encode_world_state(input_image)

    horizon = int(gt_action.shape[0])
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
                generation_horizon=horizon,
                encoded_state=encoded_state,
                seed=int(seed),
            )
            l1, l2 = per_step_errors(out["action_chunk"], gt_action)
            step_l1[mode_idx] += l1
            step_l2[mode_idx] += l2
            costs[mode_idx] = float(out["cost"])
        step_l1[mode_idx] /= len(seeds)
        step_l2[mode_idx] /= len(seeds)

    return {
        "world_feat": encoded_state.world_feat.detach().float().cpu(),
        "text_feat": pool_text_context(context, context_mask).squeeze(0).detach().float().cpu(),
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
    if step_err.shape[-2] != NUM_MODES:
        raise ValueError(
            f"`step_err` second-to-last dim must be {NUM_MODES}, got {tuple(step_err.shape)}"
        )
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
    weights = valid.unsqueeze(-2).to(step_err.dtype)
    counts = weights.sum(dim=-1)
    chunk = (step_err.float() * weights).sum(dim=-1) / counts.clamp_min(1.0)
    has_valid = counts.squeeze(-1) > 0
    chunk = torch.where(
        has_valid.unsqueeze(-1).expand_as(chunk),
        chunk,
        torch.full_like(chunk, float("inf")),
    )
    return chunk, has_valid


def select_cheapest_near_best(
    chunk_err: torch.Tensor,
    *,
    costs: Optional[torch.Tensor] = None,
    tol_abs: float = 0.0,
    tol_rel: float = 0.0,
) -> torch.Tensor:
    """Choose the cheapest mode whose error is near the best observed mode.

    This remains correct when stochastic IDM inference is worse than UNCOND.
    Invalid rows conservatively select IDM.
    """
    if float(tol_abs) < 0.0 or float(tol_rel) < 0.0:
        raise ValueError(f"tolerances must be >= 0, got {tol_abs}, {tol_rel}")
    if chunk_err.shape[-1] != NUM_MODES:
        raise ValueError(f"`chunk_err` last dim must be {NUM_MODES}, got {tuple(chunk_err.shape)}")
    err = chunk_err.float()
    finite = torch.isfinite(err)
    best = torch.where(finite, err, torch.full_like(err, float("inf"))).amin(dim=-1)
    threshold = best * (1.0 + float(tol_rel)) + float(tol_abs)
    sufficient = finite & (err <= threshold.unsqueeze(-1))

    if costs is None:
        cost = torch.tensor([0.0, 1.0], dtype=err.dtype, device=err.device)
    else:
        cost = torch.as_tensor(costs, dtype=err.dtype, device=err.device)
        if (
            cost.shape[-1] != NUM_MODES
            or not bool(torch.isfinite(cost).all())
            or bool((cost < 0).any())
        ):
            raise ValueError(
                f"costs must be finite/non-negative with last dim {NUM_MODES}, "
                f"got {tuple(cost.shape)}"
            )
    while cost.ndim < err.ndim:
        cost = cost.unsqueeze(0)
    ranked = torch.where(sufficient, cost.expand_as(err), torch.full_like(err, float("inf")))
    labels = ranked.argmin(dim=-1)
    return torch.where(sufficient.any(dim=-1), labels, torch.full_like(labels, IDM_INDEX))


def quality_metadata(chunk_err: torch.Tensor) -> dict[str, torch.Tensor]:
    """Expose absolute IDM quality and regret so failed references are visible."""
    if chunk_err.shape[-1] != NUM_MODES:
        raise ValueError(f"`chunk_err` last dim must be {NUM_MODES}, got {tuple(chunk_err.shape)}")
    finite = torch.isfinite(chunk_err)
    best = torch.where(finite, chunk_err.float(), torch.full_like(chunk_err.float(), float("inf"))).amin(-1)
    idm = chunk_err[..., IDM_INDEX].float()
    regret = torch.where(
        torch.isfinite(idm) & torch.isfinite(best),
        idm - best,
        torch.full_like(idm, float("inf")),
    )
    return {"best_err": best, "idm_err": idm, "idm_regret": regret}


def relabel_from_steps(
    step_l1: torch.Tensor,
    step_l2: torch.Tensor,
    valid_steps: torch.Tensor,
    *,
    metric: str = "l1",
    exec_horizon: Optional[int] = None,
    tol_abs: float = 0.0,
    tol_rel: float = 0.0,
    costs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if metric not in ("l1", "l2"):
        raise ValueError(f"`metric` must be 'l1' or 'l2', got {metric!r}")
    step_err = step_l1 if metric == "l1" else step_l2
    chunk_err, has_valid = chunk_errors_from_steps(step_err, valid_steps, exec_horizon=exec_horizon)
    labels = select_cheapest_near_best(
        chunk_err, costs=costs, tol_abs=tol_abs, tol_rel=tol_rel
    )
    return labels, chunk_err, has_valid


def label_distribution(labels: torch.Tensor) -> dict[str, float]:
    total = max(int(labels.numel()), 1)
    return {
        mode.value: float((labels == idx).sum().item()) / total
        for idx, mode in enumerate(MODE_ORDER)
    }


def write_label_shard(path: str, *, data: dict[str, torch.Tensor], meta: dict[str, Any]) -> None:
    missing = [key for key in SHARD_DATA_KEYS if key not in data]
    if missing:
        raise ValueError(f"label shard is missing data keys: {missing}")
    sizes = {key: int(data[key].shape[0]) for key in SHARD_DATA_KEYS}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"inconsistent leading dims across shard tensors: {sizes}")
    missing_meta = [key for key in SHARD_COMPAT_META_KEYS if key not in meta]
    if "shard_index" not in meta:
        missing_meta.append("shard_index")
    if missing_meta:
        raise ValueError(f"label shard is missing compatibility metadata: {missing_meta}")
    if data["chunk_err"].shape[-1] != NUM_MODES:
        raise ValueError(f"chunk_err must end in {NUM_MODES} modes.")
    for key in ("step_l1", "step_l2"):
        if data[key].shape[-2] != NUM_MODES:
            raise ValueError(f"{key} second-to-last dim must be {NUM_MODES}.")
    labels = data["label"].long()
    if labels.numel() and bool(((labels < 0) | (labels >= NUM_MODES)).any()):
        raise ValueError(f"labels must be in [0, {NUM_MODES}).")
    if int(data["world_feat"].shape[-1]) != int(meta["world_feat_dim"]):
        raise ValueError("world_feat shape does not match meta.world_feat_dim.")
    if int(data["text_feat"].shape[-1]) != int(meta["text_feat_dim"]):
        raise ValueError("text_feat shape does not match meta.text_feat_dim.")
    finite_keys = (
        "world_feat",
        "text_feat",
        "proprio",
        "chunk_err",
        "step_l1",
        "step_l2",
        "best_err",
        "idm_err",
        "idm_regret",
    )
    nonfinite = [
        key for key in finite_keys if not bool(torch.isfinite(data[key]).all())
    ]
    if nonfinite:
        raise ValueError(f"label shard contains non-finite tensors: {nonfinite}")
    stamped_meta = {
        **dict(meta),
        "mode_order": [mode.value for mode in MODE_ORDER],
        "compatibility_fingerprint": shard_compatibility_fingerprint(meta),
    }
    payload = {
        "version": LABEL_SHARD_VERSION,
        "meta": stamped_meta,
        "data": {key: data[key].detach().cpu() for key in SHARD_DATA_KEYS},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def resolve_shard_paths(shards: str | Sequence[str]) -> list[str]:
    patterns = [shards] if isinstance(shards, str) else list(shards)
    paths: list[str] = []
    for pattern in patterns:
        expanded = os.path.expanduser(str(pattern))
        matched = sorted(_glob.glob(expanded))
        if matched:
            paths.extend(matched)
        elif os.path.isfile(expanded):
            paths.append(expanded)
    seen: set[str] = set()
    unique = [path for path in paths if not (path in seen or seen.add(path))]
    if not unique:
        raise FileNotFoundError(f"no oracle-label shards matched: {shards!r}")
    return unique


def load_label_shards(
    shards: str | Sequence[str],
    *,
    allow_partial: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    paths = resolve_shard_paths(shards)
    datas: list[dict[str, torch.Tensor]] = []
    metas: list[dict[str, Any]] = []
    expected_modes = [mode.value for mode in MODE_ORDER]
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("version", -1)) != LABEL_SHARD_VERSION:
            raise ValueError(
                f"{path}: unsupported shard version {payload.get('version')!r}; "
                f"expected {LABEL_SHARD_VERSION}. Regenerate incompatible shards."
            )
        meta = payload["meta"]
        if list(meta.get("mode_order", [])) != expected_modes:
            raise ValueError(f"{path}: mode_order does not match {expected_modes}.")
        computed = shard_compatibility_fingerprint(meta)
        if meta.get("compatibility_fingerprint") != computed:
            raise ValueError(f"{path}: invalid or stale compatibility fingerprint.")
        missing = [key for key in SHARD_DATA_KEYS if key not in payload["data"]]
        if missing:
            raise ValueError(f"{path}: shard is missing data keys: {missing}")
        shard_data = payload["data"]
        finite_keys = (
            "world_feat", "text_feat", "proprio", "chunk_err", "step_l1",
            "step_l2", "best_err", "idm_err", "idm_regret",
        )
        nonfinite = [
            key for key in finite_keys if not bool(torch.isfinite(shard_data[key]).all())
        ]
        if nonfinite:
            raise ValueError(f"{path}: shard contains non-finite tensors: {nonfinite}")
        labels = shard_data["label"].long()
        if labels.numel() and bool(((labels < 0) | (labels >= NUM_MODES)).any()):
            raise ValueError(f"{path}: labels must be in [0, {NUM_MODES}).")
        datas.append(shard_data)
        metas.append(meta)

    reference_fp = metas[0]["compatibility_fingerprint"]
    for path, meta in zip(paths[1:], metas[1:]):
        if meta["compatibility_fingerprint"] != reference_fp:
            raise ValueError(
                f"{path}: compatibility fingerprint differs from the first shard; "
                "task/checkpoint/stats/inference/label settings cannot be mixed."
            )
    num_shards = int(metas[0]["num_shards"])
    shard_indices = [int(meta.get("shard_index", -1)) for meta in metas]
    if any(index < 0 or index >= num_shards for index in shard_indices):
        raise ValueError(
            f"shard_index values must be in [0, {num_shards}), got {shard_indices}."
        )
    if len(set(shard_indices)) != len(shard_indices):
        raise ValueError(f"duplicate shard_index values: {shard_indices}.")
    expected_indices = set(range(num_shards))
    if not allow_partial and set(shard_indices) != expected_indices:
        raise ValueError(
            f"incomplete shard set: got {sorted(shard_indices)}, expected "
            f"{sorted(expected_indices)}. Pass allow_partial=True only for deliberate analysis."
        )
    reference_shapes = {key: tuple(datas[0][key].shape[1:]) for key in SHARD_DATA_KEYS}
    for path, data in zip(paths[1:], datas[1:]):
        shapes = {key: tuple(data[key].shape[1:]) for key in SHARD_DATA_KEYS}
        if shapes != reference_shapes:
            raise ValueError(f"{path}: tensor shapes {shapes} do not match {reference_shapes}")

    merged = {key: torch.cat([data[key] for data in datas], dim=0) for key in SHARD_DATA_KEYS}
    meta = dict(metas[0])
    meta.update(
        num_shards=num_shards,
        num_loaded_shards=len(paths),
        shard_paths=paths,
        num_samples=int(merged["label"].shape[0]),
        group_split_available=bool((merged["group_id"] >= 0).all()),
    )
    return merged, meta
