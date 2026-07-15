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
from .controls import (
    IDM_CONTROL_ORDER,
    DONOR_BANK_VERSION,
    DONOR_CELL_FIELDS,
    IDMControl,
    ShuffledFutureBank,
    ShuffledFutureDonor,
    block_action_future_reads,
    coerce_idm_control,
    donor_cell,
    intervene_video_latents,
    validate_donor_metadata,
)
from .plus_manifest import (
    PLUS_MANIFEST_SCHEMA,
    PlusEpisode,
    PlusManifest,
    load_plus_manifest,
)
from .features import DEFAULT_TEXT_FEAT_DIM, TEXT_FEAT_LAYOUT, pool_text_context
from .eval_routing import explicit_eval_branch
from .provenance import (
    dual_regime_schedule_fingerprint,
    inference_solver_contract,
    inference_solver_fingerprint,
    sha256_file,
    validate_dataset_stats_fingerprint,
)
from .modes import (
    MODE_ORDER,
    NUM_MODES,
    WAMMode,
    coerce_mode,
    mode_from_index,
    mode_to_branch_steps,
    mode_to_index,
)
from .training import (
    build_optimizer_parameter_groups,
    canonicalize_uncond_weight_schedule,
    normalized_dual_regime_action_loss,
    raw_loss_gradient_statistics,
    uncond_weight_at_step,
)
from .warm_start import strict_standalone_idm_warm_start, warm_start_is_enabled
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
    "canonicalize_uncond_weight_schedule",
    "uncond_weight_at_step",
    "raw_loss_gradient_statistics",
    "build_optimizer_parameter_groups",
    "strict_standalone_idm_warm_start",
    "warm_start_is_enabled",
    "WAMModeAdapter",
    "EncodedWorldState",
    "WORLD_FEAT_LAYOUT",
    "pool_text_context",
    "DEFAULT_TEXT_FEAT_DIM",
    "TEXT_FEAT_LAYOUT",
    "explicit_eval_branch",
    "sha256_file",
    "dual_regime_schedule_fingerprint",
    "inference_solver_contract",
    "inference_solver_fingerprint",
    "validate_dataset_stats_fingerprint",
    "default_cost_table",
    "normalize_cost_table",
    "load_cost_table",
    "save_cost_table",
    "validate_cost_table",
    "ACTION_ONLY_ATTENTION_MODES",
    "validate_action_only_attention_mode",
    "IDMControl",
    "IDM_CONTROL_ORDER",
    "DONOR_BANK_VERSION",
    "DONOR_CELL_FIELDS",
    "ShuffledFutureDonor",
    "ShuffledFutureBank",
    "PLUS_MANIFEST_SCHEMA",
    "PlusEpisode",
    "PlusManifest",
    "load_plus_manifest",
    "coerce_idm_control",
    "donor_cell",
    "intervene_video_latents",
    "block_action_future_reads",
    "validate_donor_metadata",
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
