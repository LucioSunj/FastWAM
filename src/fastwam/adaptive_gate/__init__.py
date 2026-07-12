"""Adaptive-prediction gate support (fastwam side).

Mode-switched, FROZEN-WAM inference wrapper + per-mode compute cost. The RL gate
policy / env / configs live in the RLinf repo and import `WAMModeAdapter` from
here.
"""
from .cost import (
    default_cost_table,
    load_cost_table,
    normalize_cost_table,
    save_cost_table,
    validate_cost_table,
)
from .contracts import ACTION_ONLY_ATTENTION_MODES, validate_action_only_attention_mode
from .features import DEFAULT_TEXT_FEAT_DIM, TEXT_FEAT_LAYOUT, pool_text_context
from .eval_routing import explicit_eval_branch
from .provenance import sha256_file, validate_dataset_stats_fingerprint
from .modes import (
    MODE_ORDER,
    NUM_MODES,
    WAMMode,
    coerce_mode,
    mode_from_index,
    mode_to_branch_steps,
    mode_to_index,
)
from .training import normalized_dual_regime_action_loss
from .oracle import (
    LABEL_SHARD_VERSION,
    SHARD_DATA_KEYS,
    IDM_INDEX,
    chunk_errors_from_steps,
    all_mode_errors_finite,
    compose_group_id,
    compute_mode_step_errors,
    label_distribution,
    load_label_shards,
    per_step_errors,
    quality_metadata,
    relabel_from_steps,
    resolve_shard_paths,
    select_cheapest_near_best,
    shard_compatibility_fingerprint,
    write_label_shard,
)
from .wam_mode_adapter import EncodedWorldState, WAMModeAdapter, WORLD_FEAT_LAYOUT

__all__ = [
    "WAMMode",
    "MODE_ORDER",
    "NUM_MODES",
    "coerce_mode",
    "mode_from_index",
    "mode_to_index",
    "mode_to_branch_steps",
    "normalized_dual_regime_action_loss",
    "WAMModeAdapter",
    "EncodedWorldState",
    "WORLD_FEAT_LAYOUT",
    "pool_text_context",
    "DEFAULT_TEXT_FEAT_DIM",
    "TEXT_FEAT_LAYOUT",
    "explicit_eval_branch",
    "sha256_file",
    "validate_dataset_stats_fingerprint",
    "default_cost_table",
    "normalize_cost_table",
    "load_cost_table",
    "save_cost_table",
    "validate_cost_table",
    "ACTION_ONLY_ATTENTION_MODES",
    "validate_action_only_attention_mode",
    # oracle labels (M3: self-supervised SFT targets from raw VLA data)
    "IDM_INDEX",
    "LABEL_SHARD_VERSION",
    "SHARD_DATA_KEYS",
    "per_step_errors",
    "compute_mode_step_errors",
    "chunk_errors_from_steps",
    "all_mode_errors_finite",
    "compose_group_id",
    "select_cheapest_near_best",
    "quality_metadata",
    "relabel_from_steps",
    "label_distribution",
    "write_label_shard",
    "resolve_shard_paths",
    "load_label_shards",
    "shard_compatibility_fingerprint",
]
