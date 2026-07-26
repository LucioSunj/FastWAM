"""Explicit regime protocol for MoT forwards.

The fused dual-regime forward computes both conditioning regimes inside one
``MoT`` call, token-concatenated and isolated by ``MoTAttentionGroup`` spans.
Downstream stage-2 consumers (a UNCOND-gated additive LoRA, a UNCOND-only
semantic-bypass adapter) need to know, at each wrapped-submodule call, *which
regime's tokens they are transforming*. This module defines that signal.

Design: the regime is threaded through the call chain as an EXPLICIT function
argument, from ``MoTAttentionGroup.regime`` (or ``MoT.forward(active_regime=)``)
down to each wrapped submodule via :func:`regime_call`. There is deliberately
no ``current_regime()`` free function, no thread-local, and no global mutable
state — the query point IS the ``regime`` keyword a :class:`RegimeAwareModule`
receives. A module invoked outside any regime-scoped forward receives the
default ``regime=None`` ("no regime"), never a guess.

Why explicit passing is the only mechanism that stays correct:

* **Gradient checkpointing** (``use_reentrant=False``) re-executes the wrapped
  function during *backward*. ``MoT`` already captures per-call constants in
  the checkpointed closure (``_post_fn`` default arguments in ``mot.py``); the
  regime rides the same closure, so the recomputation observes the identical
  regime. Any set/unset mechanism scoped around the checkpoint boundary would
  have been torn down by the time replay runs, silently recomputing a
  *different* function than the one whose activations were saved.
* **torch.compile**: the regime is a Python string constant at trace time and
  ``isinstance`` dispatch resolves during tracing, so dynamo specializes per
  regime value (bounded by ``len(KNOWN_REGIMES)`` recompiles) instead of graph
  breaking. No mutation and no context managers are introduced anywhere on the
  traced path.

Numerical contract: with no :class:`RegimeAwareModule` attached anywhere,
:func:`regime_call` reduces to a direct ``module(*args, **kwargs)`` call — the
call sequence, operands, and dtypes are byte-identical to the pre-protocol
code, so forced-IDM bitwise parity is preserved structurally, not just
empirically.
"""
from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

REGIME_UNCOND = "uncond"
REGIME_IDM = "idm"
REGIME_JOINT = "joint"

#: Every regime name the protocol accepts. ``None`` means "no regime" and is
#: always valid as an *absence*; it is deliberately not a member of this set.
KNOWN_REGIMES = frozenset({REGIME_UNCOND, REGIME_IDM, REGIME_JOINT})


def validate_regime(regime: Optional[str], *, context: str) -> Optional[str]:
    """Validate a regime label; ``None`` (no regime) is always allowed.

    Raises:
        ValueError: If ``regime`` is neither ``None`` nor a member of
            :data:`KNOWN_REGIMES`. Unknown labels fail closed instead of being
            silently treated as "no regime".
    """
    if regime is None:
        return None
    if regime not in KNOWN_REGIMES:
        raise ValueError(
            f"{context}: unknown regime {regime!r}; expected one of "
            f"{sorted(KNOWN_REGIMES)} or None. Register new regimes in "
            "fastwam.models.wan22.regime.KNOWN_REGIMES explicitly."
        )
    return regime


class RegimeAwareModule(nn.Module):
    """Marker base class for modules that consume the regime signal.

    Subclasses implement ``forward(*args, regime=None, **kwargs)``. ``MoT``
    dispatches through :func:`regime_call` at its wrapped-submodule call sites
    (self-attention q/k/v/o projections, cross-attention, FFN), passing the
    active regime explicitly. Callers outside a regime-scoped forward — plain
    ``ActionDiT.forward``, direct module invocation, or a ``MoT`` call with
    neither tagged groups nor ``active_regime`` — leave ``regime=None``.

    Consumers must treat ``regime=None`` as "behave exactly like the wrapped
    base module": e.g. a UNCOND-gated LoRA contributes its delta only when
    ``regime == REGIME_UNCOND`` and must be an exact passthrough otherwise.
    """


def regime_call(module: nn.Module, /, *args: Any, regime: Optional[str] = None, **kwargs: Any) -> Any:
    """Invoke ``module``, forwarding ``regime`` only to regime-aware modules.

    For any other module this is exactly ``module(*args, **kwargs)`` — same
    positional/keyword operands, no extra tensor ops — which is what keeps the
    no-consumer path bitwise identical to the pre-protocol code.
    """
    if isinstance(module, RegimeAwareModule):
        return module(*args, regime=regime, **kwargs)
    return module(*args, **kwargs)


class RegimeRecorderProbe(RegimeAwareModule):
    """Test/diagnostic-only passthrough that records the regimes it observes.

    Wraps an inner module and appends ``(tag, regime)`` to :attr:`records` on
    every call, then returns ``inner(*args, **kwargs)`` unchanged — the probe
    performs no tensor arithmetic of its own, so outputs and gradients are
    bitwise identical to the unwrapped module.

    Note on gradient checkpointing: a probe wrapped inside a checkpointed
    region (cross-attention/FFN post block) records *additional* entries when
    the region is re-executed during backward. That is intentional — asserting
    that replay entries carry the same regime as the original forward is
    exactly how tests verify checkpoint-replay correctness of the protocol.

    Never ship this in a production graph; it exists for tests and offline
    diagnostics only.
    """

    def __init__(self, inner: nn.Module, *, tag: str, records: Optional[list] = None):
        super().__init__()
        self.inner = inner
        self.tag = str(tag)
        # Recording is observability, not computation: the list mutation never
        # feeds back into any tensor value, so the numeric path stays pure.
        self.records: list[tuple[str, Optional[str]]] = records if records is not None else []

    def forward(self, *args: Any, regime: Optional[str] = None, **kwargs: Any) -> Any:
        self.records.append((self.tag, regime))
        return self.inner(*args, **kwargs)
