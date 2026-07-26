"""Regime-gated additive LoRA for the UNCOND branch (stage-2 W16).

The stage-2 intervention specialises the UNCOND forward with an additive
low-rank adapter while the IDM forward stays **bitwise identical** to the
unadapted checkpoint. This module provides the three pieces downstream
work orders consume:

* :class:`RegimeGatedLoRALinear` — wraps an existing ``nn.Linear`` so that
  ``y = base(x) + (alpha/r) * B(A(x))`` **only** when the regime delivered by
  the W9 explicit-regime protocol (``fastwam.models.wan22.regime``) is
  ``"uncond"``. Any other *known* regime is an exact passthrough in which the
  LoRA branch is never even computed (no dead compute, no autograd edges).
* :func:`inject_regime_lora` — wraps matching ``nn.Linear`` submodules in a
  module tree and returns a :class:`RegimeLoRAHandle` that can enumerate the
  LoRA parameters and restore the original modules bit-identically.
* :func:`save_regime_lora_sidecar` / :func:`load_regime_lora_sidecar` —
  sidecar I/O with schema ``fastwam-regime-lora-v1``, bound to an exact
  ``parent_checkpoint_sha256`` (mirroring ``FastWAM.save_action_dit_delta``).

Fail-closed on a missing regime — a deliberate tightening of the generic
:class:`~fastwam.models.wan22.regime.RegimeAwareModule` guidance: a generic
consumer may treat ``regime=None`` as "behave like the base module", but for
the gated LoRA that would make an un-threaded call site a *silent no-op on
exactly the branch being trained*, poisoning RL training without any error.
An injected LoRA layer therefore refuses ``regime=None``: every call path
that reaches it must carry an explicit regime (``MoTAttentionGroup.regime``
or ``MoT.forward(active_regime=...)``).

Merging is permanently forbidden. There is deliberately NO API that folds
``(alpha/r) * B @ A`` into the base weight: merged weights would change the
IDM forward and cannot be un-merged from the checkpoint, destroying the
bitwise-parity contract the whole stage-2 claim rests on. Do not add one.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from fastwam.models.wan22.regime import (
    REGIME_UNCOND,
    RegimeAwareModule,
    validate_regime,
)

REGIME_LORA_SIDECAR_SCHEMA = "fastwam-regime-lora-v1"

_HEX_DIGITS = "0123456789abcdef"


def _validate_sha256(value: Any, *, what: str) -> str:
    """Mirror the ``save_action_dit_delta`` parent-sha validation exactly."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HEX_DIGITS for char in value)
    ):
        raise ValueError(
            f"{what} must be a 64-character lowercase hex SHA256 string, got "
            f"{value!r}."
        )
    return value


class RegimeGatedLoRALinear(RegimeAwareModule):
    """``nn.Linear`` wrapper adding a UNCOND-gated low-rank delta.

    Forward contract (the regime arrives via the W9 explicit protocol):

    * ``regime == "uncond"``: ``base(x) + (alpha/r) * lora_B(lora_A(x))``.
    * any other **known** regime (``"idm"``, ``"joint"``): exactly ``base(x)``.
      The LoRA branch is not executed at all — no extra tensor ops and no
      autograd graph edges touch the LoRA parameters on this path.
    * ``regime=None`` (absent) or an unknown label: raises. See the module
      docstring for why absence fails closed instead of passing through.

    ``lora_B`` is zero-initialised, so a freshly wrapped layer is an exact
    identity on the uncond path as well. Base parameters are frozen at wrap
    time (``requires_grad=False``; original flags are restored by
    :meth:`RegimeLoRAHandle.remove`) and are never mutated: the base module
    object is kept intact so removal restores it bit-identically.

    There is intentionally no merge/fold/absorb method on this class, and
    none may ever be added (see module docstring).
    """

    def __init__(self, base: nn.Linear, *, r: int, alpha: float):
        super().__init__()
        if isinstance(base, RegimeGatedLoRALinear):
            raise ValueError(
                "RegimeGatedLoRALinear cannot wrap another RegimeGatedLoRALinear; "
                "the layer is already injected."
            )
        if not isinstance(base, nn.Linear):
            raise TypeError(
                f"RegimeGatedLoRALinear requires an nn.Linear base, got {type(base).__name__}."
            )
        if not isinstance(r, int) or isinstance(r, bool) or r < 1:
            raise ValueError(f"LoRA rank `r` must be a positive integer, got {r!r}.")
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(f"LoRA `alpha` must be a finite positive number, got {alpha!r}.")

        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze the base layer; remember the original flags so removal can
        # restore the module to its exact pre-injection training state.
        self._base_requires_grad: Tuple[Tuple[str, bool], ...] = tuple(
            (name, bool(param.requires_grad)) for name, param in base.named_parameters()
        )
        for param in base.parameters():
            param.requires_grad_(False)

        dtype = base.weight.dtype
        device = base.weight.device
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device=device, dtype=dtype)
        self.lora_B.to(device=device, dtype=dtype)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor, *, regime: Optional[str] = None) -> torch.Tensor:
        regime = validate_regime(regime, context="RegimeGatedLoRALinear")
        if regime is None:
            raise ValueError(
                "RegimeGatedLoRALinear was called without a regime. The gated "
                "LoRA must only run inside a regime-scoped forward (tagged "
                "MoTAttentionGroup.regime or MoT.forward(active_regime=...)); "
                "silently skipping would make the adapter a no-op on the very "
                "branch it is training. Thread the regime to this call site or "
                "remove() the injection before running regime-free forwards."
            )
        if regime != REGIME_UNCOND:
            # Non-uncond path: the LoRA branch must not even be computed.
            return self.base(x)
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, gate=regime=='{REGIME_UNCOND}'"

    def _restore_base(self) -> nn.Linear:
        """Return the original base module with its requires_grad flags restored."""
        params = dict(self.base.named_parameters())
        for name, flag in self._base_requires_grad:
            params[name].requires_grad_(flag)
        return self.base


class RegimeLoRAHandle:
    """Handle over one :func:`inject_regime_lora` call.

    Exposes the wrapped paths, an iterator over the LoRA parameters, the
    A/B-only state dict used by the sidecar, and :meth:`remove`, which puts
    the original ``nn.Linear`` objects back bit-identically. After
    :meth:`remove` the handle is dead and every method fails closed.

    There is intentionally no merge API on the handle either.
    """

    def __init__(
        self,
        root: nn.Module,
        wrappers: Dict[str, RegimeGatedLoRALinear],
        *,
        target_spec: Tuple[str, ...],
        r: int,
        alpha: float,
    ):
        self._root = root
        self._wrappers: Dict[str, RegimeGatedLoRALinear] = dict(sorted(wrappers.items()))
        self.target_spec = target_spec
        self.r = r
        self.alpha = float(alpha)
        self._removed = False

    @property
    def wrapped_paths(self) -> Tuple[str, ...]:
        return tuple(self._wrappers)

    def _assert_alive(self, operation: str) -> None:
        if self._removed:
            raise RuntimeError(
                f"RegimeLoRAHandle.{operation}: handle is dead — remove() was "
                "already called and the original modules were restored."
            )

    def wrapper(self, path: str) -> RegimeGatedLoRALinear:
        self._assert_alive("wrapper")
        return self._wrappers[path]

    def named_lora_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        self._assert_alive("named_lora_parameters")
        for path, wrapper in self._wrappers.items():
            yield f"{path}.lora_A.weight", wrapper.lora_A.weight
            yield f"{path}.lora_B.weight", wrapper.lora_B.weight

    def lora_parameters(self) -> Iterator[nn.Parameter]:
        for _, param in self.named_lora_parameters():
            yield param

    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """A/B tensors only, detached CPU clones keyed by wrapped path."""
        self._assert_alive("lora_state_dict")
        return {
            name: param.detach().to("cpu").clone()
            for name, param in self.named_lora_parameters()
        }

    def remove(self) -> None:
        """Restore every original ``nn.Linear`` bit-identically, then die."""
        self._assert_alive("remove")
        for path, wrapper in self._wrappers.items():
            _set_submodule(self._root, path, wrapper._restore_base())
        self._removed = True


def _set_submodule(root: nn.Module, path: str, new_module: nn.Module) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if name not in parent._modules:
        raise KeyError(f"{path!r} does not name a direct child module of {parent_path!r}.")
    setattr(parent, name, new_module)


def _normalize_target_spec(target_spec: Any) -> Tuple[str, ...]:
    if isinstance(target_spec, str):
        target_spec = (target_spec,)
    if not isinstance(target_spec, Sequence) or not target_spec:
        raise ValueError(
            "target_spec must be a non-empty pattern string or sequence of "
            f"pattern strings, got {target_spec!r}."
        )
    patterns = tuple(target_spec)
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"target_spec entries must be non-empty strings, got {pattern!r}.")
    return patterns


def inject_regime_lora(
    module: nn.Module,
    target_spec: Sequence[str] | str,
    *,
    r: int,
    alpha: float,
) -> RegimeLoRAHandle:
    """Wrap every ``nn.Linear`` matching ``target_spec`` with a gated LoRA.

    ``target_spec`` is a pattern (or sequence of patterns) matched with
    ``fnmatch.fnmatchcase`` against dotted module paths from
    ``module.named_modules()`` — e.g. ``"blocks.*.self_attn.q"``.

    Fail-closed behaviour:

    * a pattern that matches no module raises (catches typos);
    * a matched module that is not an ``nn.Linear`` raises;
    * a matched module that is already a :class:`RegimeGatedLoRALinear`, or
      lives inside one (e.g. matching ``...q.base``), raises — double
      injection is never silent.

    Returns a :class:`RegimeLoRAHandle`; use ``handle.remove()`` to restore
    the original modules bit-identically. There is no merge operation.
    """
    from fnmatch import fnmatchcase

    patterns = _normalize_target_spec(target_spec)
    named = dict(module.named_modules())
    named.pop("", None)

    existing_wrapper_paths = [
        path for path, mod in named.items() if isinstance(mod, RegimeGatedLoRALinear)
    ]

    matched: List[str] = []
    seen = set()
    for pattern in patterns:
        hits = [path for path in named if fnmatchcase(path, pattern)]
        if not hits:
            raise ValueError(
                f"inject_regime_lora: pattern {pattern!r} matched no module in "
                "the target tree. Refusing to silently inject nothing."
            )
        for path in hits:
            if path in seen:
                continue
            seen.add(path)
            matched.append(path)

    wrappers: Dict[str, RegimeGatedLoRALinear] = {}
    for path in sorted(matched):
        target = named[path]
        if isinstance(target, RegimeGatedLoRALinear):
            raise ValueError(
                f"inject_regime_lora: {path!r} is already a RegimeGatedLoRALinear; "
                "double injection is forbidden."
            )
        for wrapper_path in existing_wrapper_paths:
            if path.startswith(wrapper_path + "."):
                raise ValueError(
                    f"inject_regime_lora: {path!r} lives inside the existing "
                    f"RegimeGatedLoRALinear at {wrapper_path!r}; refusing to "
                    "inject into an injected layer."
                )
        if not isinstance(target, nn.Linear):
            raise TypeError(
                f"inject_regime_lora: {path!r} is {type(target).__name__}, not "
                "nn.Linear. Point target_spec at the Linear leaves "
                "(e.g. 'blocks.*.self_attn.q')."
            )
        wrappers[path] = RegimeGatedLoRALinear(target, r=r, alpha=alpha)

    for path, wrapper in wrappers.items():
        _set_submodule(module, path, wrapper)

    return RegimeLoRAHandle(module, wrappers, target_spec=patterns, r=r, alpha=alpha)


def save_regime_lora_sidecar(
    handle: RegimeLoRAHandle,
    path,
    parent_checkpoint_sha256: str,
    step: Optional[int] = None,
) -> None:
    """Write the adapter as a ``fastwam-regime-lora-v1`` sidecar.

    The sidecar stores ONLY the LoRA A/B tensors plus the binding metadata;
    base weights never leave the parent checkpoint, and nothing here (or
    anywhere) merges the delta into them. ``parent_checkpoint_sha256`` must
    be the exact 64-hex SHA256 of the parent checkpoint file, mirroring the
    ``save_action_dit_delta`` refusal on unverifiable parents.
    """
    handle._assert_alive("save_regime_lora_sidecar")
    parent_sha = _validate_sha256(
        parent_checkpoint_sha256, what="parent_checkpoint_sha256"
    )
    if step is not None and (not isinstance(step, int) or isinstance(step, bool)):
        raise ValueError(f"step must be None or an integer, got {step!r}.")

    state_dict = handle.lora_state_dict()
    dtypes = {str(tensor.dtype) for tensor in state_dict.values()}
    torch_dtype = dtypes.pop() if len(dtypes) == 1 else "mixed"

    torch.save(
        {
            "schema": REGIME_LORA_SIDECAR_SCHEMA,
            "step": step,
            "torch_dtype": torch_dtype,
            "parent_checkpoint_sha256": parent_sha,
            "target_spec": list(handle.target_spec),
            "wrapped_paths": list(handle.wrapped_paths),
            "r": handle.r,
            "alpha": handle.alpha,
            "state_dict": state_dict,
        },
        path,
    )


def load_regime_lora_sidecar(
    handle: RegimeLoRAHandle,
    path,
    expected_parent_sha256: str,
) -> Dict[str, Any]:
    """Load a sidecar into an injected handle, failing closed on any mismatch.

    Refuses on: unknown schema, malformed or mismatched
    ``parent_checkpoint_sha256``, wrapped-path set mismatch, ``r``/``alpha``
    mismatch, missing or unexpected state keys, and per-tensor shape or dtype
    mismatch. On success copies the A/B tensors into the live wrappers and
    returns the sidecar metadata (everything except ``state_dict``).
    """
    handle._assert_alive("load_regime_lora_sidecar")
    expected_sha = _validate_sha256(
        expected_parent_sha256, what="expected_parent_sha256"
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Regime-LoRA sidecar {path!r} is not a dict payload; refusing to load."
        )
    schema = payload.get("schema")
    if schema != REGIME_LORA_SIDECAR_SCHEMA:
        raise ValueError(
            f"Regime-LoRA sidecar schema mismatch: expected "
            f"{REGIME_LORA_SIDECAR_SCHEMA!r}, got {schema!r}."
        )
    saved_sha = payload.get("parent_checkpoint_sha256")
    _validate_sha256(saved_sha, what="sidecar parent_checkpoint_sha256")
    if saved_sha != expected_sha:
        raise ValueError(
            "parent_checkpoint_sha256 mismatch: sidecar is bound to "
            f"{saved_sha}, expected {expected_sha}. Refusing to load an "
            "adapter onto a different parent checkpoint."
        )

    saved_paths = payload.get("wrapped_paths")
    if not isinstance(saved_paths, list) or set(saved_paths) != set(handle.wrapped_paths):
        raise ValueError(
            "wrapped-path set mismatch: sidecar has "
            f"{sorted(saved_paths) if isinstance(saved_paths, list) else saved_paths!r}, "
            f"handle has {sorted(handle.wrapped_paths)}."
        )
    if payload.get("r") != handle.r:
        raise ValueError(
            f"LoRA rank mismatch: sidecar r={payload.get('r')!r}, handle r={handle.r}."
        )
    if payload.get("alpha") != handle.alpha:
        raise ValueError(
            f"LoRA alpha mismatch: sidecar alpha={payload.get('alpha')!r}, "
            f"handle alpha={handle.alpha}."
        )

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Regime-LoRA sidecar has no state_dict mapping.")
    live = dict(handle.named_lora_parameters())
    saved_keys = set(state_dict)
    expected_keys = set(live)
    if saved_keys != expected_keys:
        missing = sorted(expected_keys - saved_keys)
        unexpected = sorted(saved_keys - expected_keys)
        raise ValueError(
            f"Regime-LoRA state keys mismatch: missing={missing}, unexpected={unexpected}."
        )
    for key, param in live.items():
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Regime-LoRA sidecar entry {key!r} is not a tensor.")
        if tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(
                f"Regime-LoRA shape mismatch for {key!r}: sidecar "
                f"{tuple(tensor.shape)}, module {tuple(param.shape)}."
            )
        if tensor.dtype != param.dtype:
            raise ValueError(
                f"Regime-LoRA dtype mismatch for {key!r}: sidecar {tensor.dtype}, "
                f"module {param.dtype}. A cast would break bitwise "
                "reproducibility; save and load with matching dtypes."
            )
    with torch.no_grad():
        for key, param in live.items():
            param.copy_(state_dict[key].to(device=param.device))

    return {k: v for k, v in payload.items() if k != "state_dict"}
