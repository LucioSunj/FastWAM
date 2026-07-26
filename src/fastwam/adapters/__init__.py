"""Frozen-backbone adapters for stage 2 (WS1/WS7).

Currently: the regime-gated additive LoRA that specialises the UNCOND branch
while the IDM forward stays bitwise identical. See ``regime_lora`` for the
gating, injection, and sidecar contracts — including why merging adapter
weights into base weights is permanently forbidden.
"""

from .regime_lora import (
    REGIME_LORA_SIDECAR_SCHEMA,
    RegimeGatedLoRALinear,
    RegimeLoRAHandle,
    inject_regime_lora,
    load_regime_lora_sidecar,
    save_regime_lora_sidecar,
)

__all__ = [
    "REGIME_LORA_SIDECAR_SCHEMA",
    "RegimeGatedLoRALinear",
    "RegimeLoRAHandle",
    "inject_regime_lora",
    "load_regime_lora_sidecar",
    "save_regime_lora_sidecar",
]
