"""Trainable sidecar adapters for frozen FastWAM models."""

from .merged_regime_lora import (
    FROZEN_UNCOND_ACTION_SCHEMA,
    MergedActionDiTAudit,
    load_frozen_uncond_action_artifact,
    merge_action_dit_lora_,
    save_frozen_uncond_action_artifact,
)
from .regime_lora import (
    DEFAULT_ACTION_DIT_LORA_TARGETS,
    LORA_MASTER_DTYPE,
    REGIME_LORA_SIDECAR_SCHEMA,
    ActionDiTLoRAAdapter,
    ActionLoRATargetGroup,
    BaseFreezeAudit,
    PolicyRegime,
    RegimeContext,
    RegimeLoRAConfig,
    RegimeLoRALinear,
    discover_action_dit_lora_targets,
    inject_action_dit_lora,
    sha256_file,
)

__all__ = [
    "DEFAULT_ACTION_DIT_LORA_TARGETS",
    "FROZEN_UNCOND_ACTION_SCHEMA",
    "LORA_MASTER_DTYPE",
    "REGIME_LORA_SIDECAR_SCHEMA",
    "ActionDiTLoRAAdapter",
    "ActionLoRATargetGroup",
    "BaseFreezeAudit",
    "MergedActionDiTAudit",
    "PolicyRegime",
    "RegimeContext",
    "RegimeLoRAConfig",
    "RegimeLoRALinear",
    "discover_action_dit_lora_targets",
    "inject_action_dit_lora",
    "load_frozen_uncond_action_artifact",
    "merge_action_dit_lora_",
    "save_frozen_uncond_action_artifact",
    "sha256_file",
]
