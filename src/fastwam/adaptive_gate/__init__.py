"""Adaptive-prediction gate support (fastwam side).

Mode-switched, FROZEN-WAM inference wrapper + per-mode compute cost. The RL gate
policy / env / configs live in the RLinf repo and import `WAMModeAdapter` from
here.
"""
from .cost import default_cost_table, load_cost_table, normalize_cost_table, save_cost_table
from .modes import (
    MODE_ORDER,
    NUM_MODES,
    WAMMode,
    coerce_mode,
    future_branch_for,
    mode_to_branch_step_counts,
    mode_from_index,
    mode_to_branch_steps,
    mode_to_index,
)
from .oracle import (
    FULL_INDEX,
    LABEL_SHARD_VERSION,
    SHARD_DATA_KEYS,
    chunk_errors_from_steps,
    compute_mode_step_errors,
    label_distribution,
    load_label_shards,
    per_step_errors,
    relabel_from_steps,
    resolve_shard_paths,
    select_cheapest_sufficient,
    write_label_shard,
)
from .wam_mode_adapter import WAMModeAdapter

__all__ = [
    "WAMMode",
    "MODE_ORDER",
    "NUM_MODES",
    "coerce_mode",
    "mode_from_index",
    "mode_to_index",
    "future_branch_for",
    "mode_to_branch_step_counts",
    "mode_to_branch_steps",
    "WAMModeAdapter",
    "default_cost_table",
    "normalize_cost_table",
    "load_cost_table",
    "save_cost_table",
    # oracle labels (M3: self-supervised SFT targets from raw VLA data)
    "FULL_INDEX",
    "LABEL_SHARD_VERSION",
    "SHARD_DATA_KEYS",
    "per_step_errors",
    "compute_mode_step_errors",
    "chunk_errors_from_steps",
    "select_cheapest_sufficient",
    "relabel_from_steps",
    "label_distribution",
    "write_label_shard",
    "resolve_shard_paths",
    "load_label_shards",
]
