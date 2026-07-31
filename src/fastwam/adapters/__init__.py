"""Trainable sidecar adapters for frozen FastWAM models."""

from .regime_lora import (
    DEFAULT_ACTION_DIT_LORA_TARGETS,
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
    "REGIME_LORA_SIDECAR_SCHEMA",
    "ActionDiTLoRAAdapter",
    "ActionLoRATargetGroup",
    "BaseFreezeAudit",
    "PolicyRegime",
    "RegimeContext",
    "RegimeLoRAConfig",
    "RegimeLoRALinear",
    "discover_action_dit_lora_targets",
    "inject_action_dit_lora",
    "sha256_file",
]
