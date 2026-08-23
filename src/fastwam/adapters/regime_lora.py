"""UNCOND-only LoRA adapters for a frozen ActionDiT.

The regime is carried by an instance-scoped :class:`RegimeContext`. It is not a
module-level flag, and its context manager always restores the previous value.
This lets one ActionDiT serve forced IDM and UNCOND calls without allowing a
failed or nested call to leak its regime into the next forward.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

REGIME_LORA_SIDECAR_SCHEMA = "fastwam-regime-lora-v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PolicyRegime(str, Enum):
    """Execution regime for the shared FastWAM action expert."""

    IDM = "idm"
    UNCOND = "uncond"

    @classmethod
    def parse(cls, value: PolicyRegime | str) -> PolicyRegime:
        """Return a validated regime value."""

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as error:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown policy regime {value!r}; expected one of: {allowed}"
            ) from error


class RegimeContext:
    """Concurrency-safe, instance-scoped policy-regime context.

    IDM is the fail-safe default because it must never activate the UNCOND
    adapter accidentally. ``ContextVar`` isolates concurrent tasks and threads,
    while the returned token gives nested scopes exact stack semantics.
    """

    def __init__(self, default: PolicyRegime | str = PolicyRegime.IDM) -> None:
        self._default = PolicyRegime.parse(default)
        self._value: ContextVar[PolicyRegime] = ContextVar(
            f"fastwam_policy_regime_{id(self)}",
            default=self._default,
        )

    @property
    def current(self) -> PolicyRegime:
        """Return the regime active in the current execution context."""

        return self._value.get()

    def __deepcopy__(self, memo: dict[int, Any]) -> RegimeContext:
        """Create a fresh context while preserving shared references in a model copy."""

        existing = memo.get(id(self))
        if existing is not None:
            return existing
        copied = type(self)(self._default)
        memo[id(self)] = copied
        return copied

    @contextmanager
    def use(self, regime: PolicyRegime | str) -> Iterator[PolicyRegime]:
        """Temporarily select a regime and restore it even if the call fails."""

        selected = PolicyRegime.parse(regime)
        token = self._value.set(selected)
        try:
            yield selected
        finally:
            self._value.reset(token)


class ActionLoRATargetGroup(str, Enum):
    """Configurable groups of ActionDiT linear projections."""

    SELF_ATTENTION_QKVO = "self_attention_qkvo"
    CROSS_ATTENTION_QKVO = "cross_attention_qkvo"
    FFN = "ffn"

    @classmethod
    def parse(cls, value: ActionLoRATargetGroup | str) -> ActionLoRATargetGroup:
        """Return a validated target group."""

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as error:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown ActionDiT LoRA target {value!r}; expected: {allowed}"
            ) from error


DEFAULT_ACTION_DIT_LORA_TARGETS = (
    ActionLoRATargetGroup.SELF_ATTENTION_QKVO,
    ActionLoRATargetGroup.CROSS_ATTENTION_QKVO,
    ActionLoRATargetGroup.FFN,
)


@dataclass(frozen=True)
class RegimeLoRAConfig:
    """Configuration for an ActionDiT regime-gated LoRA."""

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
        if self.rank <= 0:
            raise ValueError(f"`rank` must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"`alpha` must be positive, got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {self.dropout}")
        if not groups:
            raise ValueError("At least one ActionDiT LoRA target group is required.")
        if len(set(groups)) != len(groups):
            raise ValueError(
                f"Duplicate ActionDiT LoRA target groups are not allowed: {groups}"
            )
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "target_groups", groups)


# Trainable LoRA factors are kept in FP32 regardless of the frozen base dtype.
# A BF16 parameter silently discards any optimizer update smaller than half its
# unit-in-last-place, and the Adam step for these factors is one to two orders
# of magnitude below that threshold, so a BF16 adapter never leaves its
# initialization. See `docs/BF16_PARAMETER_UPDATE_LOSS.md`.
LORA_MASTER_DTYPE = torch.float32


class RegimeLoRALinear(nn.Linear):
    """An additive LoRA projection that is active only in UNCOND.

    The original ``weight`` and ``bias`` Parameter objects are retained, so base
    checkpoint key names do not change after injection. The LoRA factors are
    FP32 master weights and are cast to the base dtype at use, so the delta is
    computed in the frozen base precision while the optimizer writes FP32.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        regime_context: RegimeContext,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        if isinstance(base, RegimeLoRALinear):
            raise TypeError("Refusing to wrap an already adapted linear layer.")
        if rank <= 0:
            raise ValueError(f"`rank` must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"`alpha` must be positive, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}")
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
        self.regime_context = regime_context
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
        self.reset_lora_parameters()
        self.train(base.training)

    def reset_lora_parameters(self) -> None:
        """Initialize A conventionally and B to an exact zero delta."""

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base projection and an UNCOND-only LoRA delta."""

        output = F.linear(input, self.weight, self.bias)
        if self.regime_context.current is not PolicyRegime.UNCOND:
            return output
        # Cast the FP32 master factors down at use. The delta therefore keeps
        # the numerics of a fully base-dtype adapter while gradients still
        # accumulate into the FP32 leaves the optimizer owns.
        hidden = F.linear(self.lora_dropout(input), self.lora_A.to(dtype=input.dtype))
        delta = F.linear(hidden, self.lora_B.to(dtype=input.dtype))
        return output + delta * self.scaling


_TARGET_PATTERNS = {
    ActionLoRATargetGroup.SELF_ATTENTION_QKVO: re.compile(
        r"^blocks\.(?P<block>\d+)\.self_attn\.(?P<projection>q|k|v|o)$"
    ),
    ActionLoRATargetGroup.CROSS_ATTENTION_QKVO: re.compile(
        r"^blocks\.(?P<block>\d+)\.cross_attn\.(?P<projection>q|k|v|o)$"
    ),
    ActionLoRATargetGroup.FFN: re.compile(
        r"^blocks\.(?P<block>\d+)\.ffn\.(?P<projection>0|2)$"
    ),
}


def _expected_target_names(
    action_dit: nn.Module,
    groups: Sequence[ActionLoRATargetGroup],
) -> set[str]:
    blocks = getattr(action_dit, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError(
            "ActionDiT LoRA expects `action_dit.blocks` to be an nn.ModuleList."
        )

    expected: set[str] = set()
    for index in range(len(blocks)):
        if ActionLoRATargetGroup.SELF_ATTENTION_QKVO in groups:
            expected.update(
                f"blocks.{index}.self_attn.{projection}"
                for projection in ("q", "k", "v", "o")
            )
        if ActionLoRATargetGroup.CROSS_ATTENTION_QKVO in groups:
            expected.update(
                f"blocks.{index}.cross_attn.{projection}"
                for projection in ("q", "k", "v", "o")
            )
        if ActionLoRATargetGroup.FFN in groups:
            expected.update((f"blocks.{index}.ffn.0", f"blocks.{index}.ffn.2"))
    return expected


def discover_action_dit_lora_targets(
    action_dit: nn.Module,
    target_groups: Sequence[
        ActionLoRATargetGroup | str
    ] = DEFAULT_ACTION_DIT_LORA_TARGETS,
    *,
    strict: bool = True,
) -> tuple[str, ...]:
    """Discover configured ActionDiT block projections by structural name."""

    groups = tuple(ActionLoRATargetGroup.parse(group) for group in target_groups)
    expected = _expected_target_names(action_dit, groups)
    discovered: set[str] = set()
    for name, module in action_dit.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(_TARGET_PATTERNS[group].fullmatch(name) for group in groups):
            discovered.add(name)

    if strict and discovered != expected:
        missing = sorted(expected - discovered)
        unexpected = sorted(discovered - expected)
        raise ValueError(
            "ActionDiT LoRA target discovery did not match the current block contract. "
            f"missing={missing}, unexpected={unexpected}"
        )
    return tuple(sorted(discovered))


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if child_name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


@dataclass(frozen=True)
class BaseFreezeAudit:
    """Result of auditing trainable base and adapter parameters."""

    trainable_base: tuple[str, ...]
    trainable_lora: tuple[str, ...]
    frozen_lora: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """Whether only adapter parameters are trainable."""

        return (
            not self.trainable_base
            and not self.frozen_lora
            and bool(self.trainable_lora)
        )

    def assert_valid(self) -> None:
        """Raise with the precise parameter names when the freeze contract fails."""

        if not self.valid:
            raise RuntimeError(
                "ActionDiT freeze audit failed: "
                f"trainable_base={list(self.trainable_base)}, "
                f"frozen_lora={list(self.frozen_lora)}, "
                f"trainable_lora={list(self.trainable_lora)}"
            )


class ActionDiTLoRAAdapter:
    """Controller and serialization surface for injected ActionDiT adapters."""

    def __init__(
        self,
        action_dit: nn.Module,
        *,
        config: RegimeLoRAConfig,
        regime_context: RegimeContext,
        target_names: Sequence[str],
    ) -> None:
        self.action_dit = action_dit
        self.config = config
        self.regime_context = regime_context
        self.target_names = tuple(target_names)
        self._replay_reference: dict[str, torch.Tensor] | None = None
        self._replay_reference_actor_version: int | None = None

    def use_regime(self, regime: PolicyRegime | str):
        """Return a scoped regime context for a model forward."""

        return self.regime_context.use(regime)

    @staticmethod
    def _unwrap_adapted_linear(module: nn.Module) -> RegimeLoRALinear:
        """Find an adapted projection through transparent module wrappers."""

        original_type = type(module).__name__
        visited: set[int] = set()
        while id(module) not in visited:
            if isinstance(module, RegimeLoRALinear):
                return module
            visited.add(id(module))
            wrapped = getattr(module, "_fsdp_wrapped_module", None)
            if wrapped is None:
                wrapped = getattr(module, "module", None)
            if not isinstance(wrapped, nn.Module):
                break
            module = wrapped
        raise RuntimeError(
            "Expected an adapted linear through optional wrappers, found "
            f"{original_type}."
        )

    def iter_adapted_linears(self) -> Iterator[tuple[str, RegimeLoRALinear]]:
        """Yield every injected linear layer in deterministic name order."""

        for name in self.target_names:
            module = self.action_dit.get_submodule(name)
            yield name, self._unwrap_adapted_linear(module)

    def named_lora_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield adapter-only parameters, suitable for an optimizer group."""

        for name, module in self.iter_adapted_linears():
            yield f"{name}.lora_A", module.lora_A
            yield f"{name}.lora_B", module.lora_B

    def lora_parameters(self) -> Iterator[nn.Parameter]:
        """Yield adapter-only parameters."""

        for _, parameter in self.named_lora_parameters():
            yield parameter

    def freeze_base(self) -> BaseFreezeAudit:
        """Freeze all ActionDiT parameters and re-enable only LoRA A/B."""

        for parameter in self.action_dit.parameters():
            parameter.requires_grad_(False)
        for parameter in self.lora_parameters():
            parameter.requires_grad_(True)
        audit = self.audit_freeze()
        audit.assert_valid()
        return audit

    def audit_freeze(self) -> BaseFreezeAudit:
        """Report violations of the frozen-base/trainable-adapter contract."""

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
        """Return an adapter-only state dict without any base weights."""

        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_lora_parameters()
        }

    def capture_replay_reference(self, *, actor_version: int) -> None:
        """Snapshot the behavior LoRA used to reconstruct Gate observations."""

        if actor_version < 0:
            raise ValueError("Replay-reference actor version must be non-negative.")
        self._replay_reference = {
            name: parameter.detach().clone()
            for name, parameter in self.named_lora_parameters()
        }
        self._replay_reference_actor_version = int(actor_version)

    @contextmanager
    def use_replay_reference(self, *, actor_version: int) -> Iterator[None]:
        """Temporarily restore behavior LoRA for no-grad K/V recomputation."""

        if self._replay_reference is None:
            raise RuntimeError("No behavior LoRA replay reference was captured.")
        if actor_version != self._replay_reference_actor_version:
            raise ValueError(
                "Gate K/V replay actor version mismatch: "
                f"expected {self._replay_reference_actor_version}, "
                f"got {actor_version}."
            )
        current = {
            name: parameter.detach().clone()
            for name, parameter in self.named_lora_parameters()
        }
        try:
            self.load_lora_state_dict(self._replay_reference, strict=True)
            yield
        finally:
            self.load_lora_state_dict(current, strict=True)

    def load_lora_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> None:
        """Load adapter-only tensors and validate keys and shapes."""

        parameters = dict(self.named_lora_parameters())
        provided = set(state_dict)
        expected = set(parameters)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if strict and (missing or unexpected):
            raise ValueError(
                f"LoRA state key mismatch: missing={missing}, unexpected={unexpected}"
            )

        with torch.no_grad():
            for name in sorted(expected & provided):
                value = state_dict[name]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"LoRA state {name!r} must be a tensor, got {type(value)}"
                    )
                target = parameters[name]
                if value.shape != target.shape:
                    raise ValueError(
                        f"LoRA state shape mismatch for {name}: "
                        f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                    )
                target.copy_(value.to(device=target.device, dtype=target.dtype))

    def sidecar_metadata(
        self,
        *,
        parent_checkpoint_sha256: str,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build versioned metadata bound to the exact parent checkpoint."""

        parent_hash = _validate_sha256(parent_checkpoint_sha256)
        return {
            "schema": REGIME_LORA_SIDECAR_SCHEMA,
            "parent_checkpoint_sha256": parent_hash,
            "active_regime": PolicyRegime.UNCOND.value,
            "rank": self.config.rank,
            "alpha": self.config.alpha,
            "dropout": self.config.dropout,
            "target_groups": [group.value for group in self.config.target_groups],
            "target_names": list(self.target_names),
            "extra": dict(extra_metadata or {}),
        }

    def save_sidecar(
        self,
        path: str | os.PathLike[str],
        *,
        parent_checkpoint_sha256: str,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically save adapter tensors and parent-bound metadata."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.sidecar_metadata(
                parent_checkpoint_sha256=parent_checkpoint_sha256,
                extra_metadata=extra_metadata,
            ),
            "state_dict": self.lora_state_dict(),
        }
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def load_sidecar(
        self,
        path: str | os.PathLike[str],
        *,
        expected_parent_checkpoint_sha256: str,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load a sidecar after validating schema, parent, config, and targets."""

        expected_hash = _validate_sha256(expected_parent_checkpoint_sha256)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid LoRA sidecar payload type: {type(payload)}")
        metadata = payload.get("metadata")
        state_dict = payload.get("state_dict")
        if not isinstance(metadata, dict) or not isinstance(state_dict, dict):
            raise TypeError(
                "LoRA sidecar requires dict `metadata` and `state_dict` fields."
            )
        if metadata.get("schema") != REGIME_LORA_SIDECAR_SCHEMA:
            raise ValueError(
                f"Unsupported LoRA sidecar schema {metadata.get('schema')!r}; "
                f"expected {REGIME_LORA_SIDECAR_SCHEMA!r}"
            )
        if metadata.get("parent_checkpoint_sha256") != expected_hash:
            raise ValueError(
                "LoRA sidecar parent checkpoint mismatch: "
                f"expected {expected_hash}, got {metadata.get('parent_checkpoint_sha256')}"
            )

        expected_contract = {
            "active_regime": PolicyRegime.UNCOND.value,
            "rank": self.config.rank,
            "alpha": self.config.alpha,
            "dropout": self.config.dropout,
            "target_groups": [group.value for group in self.config.target_groups],
            "target_names": list(self.target_names),
        }
        mismatches = {
            key: (expected, metadata.get(key))
            for key, expected in expected_contract.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"LoRA sidecar adapter contract mismatch: {mismatches}")

        self.load_lora_state_dict(state_dict, strict=strict)
        return metadata


def inject_action_dit_lora(
    action_dit: nn.Module,
    config: RegimeLoRAConfig | None = None,
    *,
    regime_context: RegimeContext | None = None,
) -> ActionDiTLoRAAdapter:
    """Inject UNCOND-only LoRA projections into an initialized ActionDiT."""

    resolved_config = config or RegimeLoRAConfig()
    context = regime_context or RegimeContext()
    target_names = discover_action_dit_lora_targets(
        action_dit,
        resolved_config.target_groups,
        strict=resolved_config.strict_target_discovery,
    )
    if not target_names:
        raise ValueError("No ActionDiT LoRA targets were discovered.")
    if any(
        isinstance(action_dit.get_submodule(name), RegimeLoRALinear)
        for name in target_names
    ):
        raise ValueError("ActionDiT already contains regime-gated LoRA layers.")

    for name in target_names:
        base = action_dit.get_submodule(name)
        if not isinstance(base, nn.Linear):
            raise TypeError(
                f"ActionDiT target {name} must be nn.Linear, got {type(base)}"
            )
        adapted = RegimeLoRALinear(
            base,
            regime_context=context,
            rank=resolved_config.rank,
            alpha=resolved_config.alpha,
            dropout=resolved_config.dropout,
        )
        _replace_submodule(action_dit, name, adapted)

    adapter = ActionDiTLoRAAdapter(
        action_dit,
        config=resolved_config,
        regime_context=context,
        target_names=target_names,
    )
    if resolved_config.freeze_base:
        adapter.freeze_base()
    return adapter


def _validate_sha256(value: str) -> str:
    normalized = str(value).lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(
            "`parent_checkpoint_sha256` must be exactly 64 hexadecimal characters, "
            f"got {value!r}"
        )
    return normalized


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a checkpoint or other parent artifact."""

    if chunk_size <= 0:
        raise ValueError(f"`chunk_size` must be positive, got {chunk_size}")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
