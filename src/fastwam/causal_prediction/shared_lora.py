"""Always-active ActionDiT LoRA shared by every causal compute mode.

The existing regime-gated adapter is deliberately left unchanged. This module
uses the same target-discovery contract but owns a separate checkpoint schema
and applies one parameter set in C0, C1, and C2.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fastwam.adapters.regime_lora import (
    DEFAULT_ACTION_DIT_LORA_TARGETS,
    ActionLoRATargetGroup,
    BaseFreezeAudit,
    RegimeLoRAConfig,
    LORA_MASTER_DTYPE,
    discover_action_dit_lora_targets,
)

from .contracts import CAUSAL_POLICY_CHECKPOINT_SCHEMA

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SharedLoRAContext:
    """Concurrency-safe switch used only for frozen-teacher/parity calls."""

    def __init__(self) -> None:
        self._enabled: ContextVar[bool] = ContextVar(
            f"fastwam_shared_lora_{id(self)}",
            default=True,
        )

    @property
    def enabled(self) -> bool:
        """Whether the shared causal delta is active in this call context."""

        return self._enabled.get()

    def __deepcopy__(self, memo: dict[int, Any]) -> SharedLoRAContext:
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        copied = type(self)()
        memo[id(self)] = copied
        return copied

    @contextmanager
    def base_only(self) -> Iterator[None]:
        """Disable the delta temporarily for teacher and parent-parity audits."""

        token = self._enabled.set(False)
        try:
            yield
        finally:
            self._enabled.reset(token)


class SharedLoRALinear(nn.Linear):
    """An additive LoRA projection active in all causal compute modes.

    The LoRA factors are FP32 master weights cast to the base dtype at use. A
    reduced-precision factor would discard every optimizer step below half its
    unit-in-last-place; see `docs/BF16_PARAMETER_UPDATE_LOSS.md`.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        shared_context: SharedLoRAContext,
    ) -> None:
        if isinstance(base, SharedLoRALinear):
            raise TypeError("Refusing to wrap an already shared-adapted layer.")
        if rank <= 0 or alpha <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("Invalid shared LoRA rank, alpha, or dropout.")
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.weight = base.weight
        self.bias = base.bias
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.shared_context = shared_context
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                self.in_features,
                device=self.weight.device,
                dtype=LORA_MASTER_DTYPE,
            )
        )
        self.lora_B = nn.Parameter(
            torch.empty(
                self.out_features,
                self.rank,
                device=self.weight.device,
                dtype=LORA_MASTER_DTYPE,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.train(base.training)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base projection plus the common LoRA delta."""

        output = F.linear(input, self.weight, self.bias)
        if not self.shared_context.enabled:
            return output
        # Cast the FP32 master factors down at use so the delta keeps the
        # numerics of a base-dtype adapter while gradients accumulate in FP32.
        hidden = F.linear(self.lora_dropout(input), self.lora_A.to(dtype=input.dtype))
        return (
            output
            + F.linear(hidden, self.lora_B.to(dtype=input.dtype)) * self.scaling
        )


@dataclass(frozen=True)
class SharedLoRAConfig:
    """Configuration for the causal shared ActionDiT LoRA."""

    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_groups: tuple[ActionLoRATargetGroup, ...] = DEFAULT_ACTION_DIT_LORA_TARGETS
    freeze_base: bool = True
    strict_target_discovery: bool = True

    def __post_init__(self) -> None:
        groups = tuple(
            ActionLoRATargetGroup.parse(group) for group in self.target_groups
        )
        validated = RegimeLoRAConfig(
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target_groups=groups,
            freeze_base=self.freeze_base,
            strict_target_discovery=self.strict_target_discovery,
        )
        object.__setattr__(self, "rank", validated.rank)
        object.__setattr__(self, "alpha", validated.alpha)
        object.__setattr__(self, "dropout", validated.dropout)
        object.__setattr__(self, "target_groups", validated.target_groups)

    def as_regime_config(self) -> RegimeLoRAConfig:
        """Reuse the established target validation without its route semantics."""

        return RegimeLoRAConfig(
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target_groups=self.target_groups,
            freeze_base=self.freeze_base,
            strict_target_discovery=self.strict_target_discovery,
        )


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if child_name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


class SharedActionDiTLoRAAdapter:
    """Ownership, freeze audit, and serialization for the shared adapter."""

    def __init__(
        self,
        action_dit: nn.Module,
        *,
        config: SharedLoRAConfig,
        target_names: Sequence[str],
        shared_context: SharedLoRAContext,
    ) -> None:
        self.action_dit = action_dit
        self.config = config
        self.target_names = tuple(target_names)
        self.shared_context = shared_context

    def base_only(self):
        """Return the narrow frozen-teacher/parity context."""

        return self.shared_context.base_only()

    def iter_adapted_linears(self) -> Iterator[tuple[str, SharedLoRALinear]]:
        """Yield all shared projections in deterministic name order."""

        for name in self.target_names:
            module = self.action_dit.get_submodule(name)
            if not isinstance(module, SharedLoRALinear):
                raise TypeError(
                    f"Expected SharedLoRALinear at {name!r}, got {type(module)}."
                )
            yield name, module

    def named_lora_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield only the shared adapter parameters."""

        for name, module in self.iter_adapted_linears():
            yield f"{name}.lora_A", module.lora_A
            yield f"{name}.lora_B", module.lora_B

    def lora_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only the shared adapter parameters."""

        for _, parameter in self.named_lora_parameters():
            yield parameter

    def freeze_base(self) -> BaseFreezeAudit:
        """Freeze the parent and enable exactly the shared LoRA tensors."""

        for parameter in self.action_dit.parameters():
            parameter.requires_grad_(False)
        for parameter in self.lora_parameters():
            parameter.requires_grad_(True)
        audit = self.audit_freeze()
        audit.assert_valid()
        return audit

    def audit_freeze(self) -> BaseFreezeAudit:
        """Report all base and LoRA ownership violations."""

        trainable_base: list[str] = []
        trainable_lora: list[str] = []
        frozen_lora: list[str] = []
        for name, parameter in self.action_dit.named_parameters():
            is_lora = name.endswith((".lora_A", ".lora_B"))
            if is_lora and parameter.requires_grad:
                trainable_lora.append(name)
            elif is_lora:
                frozen_lora.append(name)
            elif parameter.requires_grad:
                trainable_base.append(name)
        return BaseFreezeAudit(
            trainable_base=tuple(trainable_base),
            trainable_lora=tuple(trainable_lora),
            frozen_lora=tuple(frozen_lora),
        )

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Return adapter-only tensors; frozen parent tensors are impossible here."""

        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_lora_parameters()
        }

    def load_lora_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> None:
        """Load shared tensors after exact key and shape validation."""

        parameters = dict(self.named_lora_parameters())
        expected = set(parameters)
        provided = set(state_dict)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if strict and (missing or unexpected):
            raise ValueError(
                f"Shared LoRA key mismatch: missing={missing}, unexpected={unexpected}"
            )
        with torch.no_grad():
            for name in sorted(expected & provided):
                value = state_dict[name]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Shared LoRA state {name!r} must be a tensor.")
                target = parameters[name]
                if value.shape != target.shape:
                    raise ValueError(
                        f"Shared LoRA shape mismatch for {name}: expected "
                        f"{tuple(target.shape)}, got {tuple(value.shape)}"
                    )
                target.copy_(value.to(device=target.device, dtype=target.dtype))

    def metadata(
        self,
        *,
        parent_checkpoint_sha256: str,
        statistics_sha256: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the exact parent-bound shared-policy contract."""

        parent_hash = _validate_sha256(parent_checkpoint_sha256)
        statistics_hash = _validate_sha256(statistics_sha256)
        return {
            "schema": CAUSAL_POLICY_CHECKPOINT_SCHEMA,
            "parent_checkpoint_sha256": parent_hash,
            "statistics_sha256": statistics_hash,
            "active_modes": ["c0_current", "c2_full"],
            "rank": int(self.config.rank),
            "alpha": float(self.config.alpha),
            "dropout": float(self.config.dropout),
            "target_groups": [group.value for group in self.config.target_groups],
            "target_names": list(self.target_names),
            "extra": dict(extra or {}),
        }

    def save_sidecar(
        self,
        path: str | os.PathLike[str],
        *,
        parent_checkpoint_sha256: str,
        statistics_sha256: str,
        trainer_state: Mapping[str, Any],
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Save a resumable shared-policy sidecar without parent tensors."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata(
                parent_checkpoint_sha256=parent_checkpoint_sha256,
                statistics_sha256=statistics_sha256,
                extra=extra,
            ),
            "state_dict": self.lora_state_dict(),
            "trainer_state": dict(trainer_state),
        }
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def inject_shared_action_dit_lora(
    action_dit: nn.Module,
    config: SharedLoRAConfig | None = None,
) -> SharedActionDiTLoRAAdapter:
    """Inject one always-active LoRA parameter set into ActionDiT."""

    resolved = config or SharedLoRAConfig()
    shared_context = SharedLoRAContext()
    validated = resolved.as_regime_config()
    target_names = discover_action_dit_lora_targets(
        action_dit,
        validated.target_groups,
        strict=validated.strict_target_discovery,
    )
    if not target_names:
        raise ValueError("No ActionDiT shared LoRA targets were discovered.")
    for name in target_names:
        base = action_dit.get_submodule(name)
        if not isinstance(base, nn.Linear) or isinstance(base, (SharedLoRALinear,)):
            raise TypeError(
                f"Shared LoRA target {name} must be an unadapted nn.Linear, "
                f"got {type(base)}."
            )
        _replace_submodule(
            action_dit,
            name,
            SharedLoRALinear(
                base,
                rank=resolved.rank,
                alpha=resolved.alpha,
                dropout=resolved.dropout,
                shared_context=shared_context,
            ),
        )
    adapter = SharedActionDiTLoRAAdapter(
        action_dit,
        config=resolved,
        target_names=target_names,
        shared_context=shared_context,
    )
    if resolved.freeze_base:
        adapter.freeze_base()
    return adapter


def _validate_sha256(value: str) -> str:
    normalized = str(value).lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("Checkpoint identities must be 64 lowercase hex characters.")
    return normalized
