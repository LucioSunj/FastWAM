"""Read-only activation taps for the E7 three-point probe.

Decision point DP2 does not ask "H1 or H2"; it asks *where* the information is
lost under texture shift. Answering that needs the same linear read-out fitted at
three loci. AUC collapsing at the VAE latent means the encoder itself lost it ->
an external semantic encoder is required (route C2). At the video-expert hidden
state -> a video-side UNCOND-gated LoRA (route C1). Nowhere -> the information
survives and route A suffices.

**Not every locus is reachable by a forward hook**, and getting this wrong is easy
because the obvious names are methods rather than submodules. Statically verified
against FastWAM `ac7d376`:

===========================  =====================================================
locus                        how to obtain it
===========================  =====================================================
(a) VAE latent               **not hookable via the adapter.** `WAMModeAdapter` is
                             a plain class, not an `nn.Module`
                             (`adaptive_gate/wam_mode_adapter.py:36`), so
                             `encode_world_state` (`:325`) carries no hooks. Either
                             call it and keep its returned
                             `EncodedWorldState.first_frame_latents` directly -- no
                             taps needed, `probe_taps` accepts plain arrays -- or
                             hook the `model.vae` submodule
                             (`models/wan22/fastwam.py:56`).
(b) video-expert hidden      `video_expert.pre_dit` is a **method**
                             (`wan_video_dit.py:509`) and cannot be hooked. Hook
                             `video_expert.blocks.N` instead
                             (`nn.ModuleList`, `wan_video_dit.py:381`).
(c) action-expert readout    `action_expert.post_dit` is a **method**
                             (`action_dit.py:301`); it is a thin wrapper over
                             `self.head`. Hook `action_expert.head`
                             (`nn.Linear`, `action_dit.py:98`) with
                             ``site="input"`` -- that input *is* the readout state.
===========================  =====================================================

`CANDIDATE_FASTWAM_TAPS` below encodes this. It is derived from the class
definitions only: **no real forward pass has ever been run through it**, so the
firing order, tensor shapes and correct `feature_dim` values are unverified. The
measurement work order must check them against a live model before trusting any
number.

**This module must never modify `models/wan22/`.** Extraction is done purely with
`torch.nn.Module.register_forward_hook` / `register_forward_pre_hook`, for two
reasons: another development lane owns those files, and a hook cannot perturb the
numerical path the way an inline edit could. A probe that changed the thing it
measures would be worthless.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

__all__ = [
    "CANDIDATE_FASTWAM_TAPS",
    "TapSpec",
    "ActivationTaps",
    "resolve_module",
]


@dataclass(frozen=True)
class TapSpec:
    """One named extraction point.

    Args:
        name: identifier used to retrieve the captured activations. Must be
            unique within an `ActivationTaps`.
        module_path: dotted attribute path from the root module. Numeric
            components index into `nn.Sequential` / `nn.ModuleList`, e.g.
            ``"video_expert.blocks.11"``. An empty string taps the root itself.
        site: ``"output"`` (forward hook) or ``"input"`` (forward pre-hook).
        select: how to pick a tensor out of a non-tensor payload. ``None``
            requires the payload to already be a tensor; an ``int`` indexes a
            tuple/list; a ``str`` keys a mapping. MoT and the video expert both
            return dicts, so this is load-bearing rather than decorative.
        feature_dim: which axis of the captured tensor carries features. Consumed
            downstream by `fastwam.diagnostics.probe.pool_activation`; recorded
            here so the tap definition stays the single source of truth. For
            ``[B, T, D]`` token streams this is ``-1``; for ``[B, C, T, H, W]``
            video latents it is ``1``.
    """

    name: str
    module_path: str
    site: str = "output"
    select: int | str | None = None
    feature_dim: int = -1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TapSpec.name must be a non-empty string.")
        if self.site not in ("output", "input"):
            raise ValueError(
                f"TapSpec.site must be 'output' or 'input', got {self.site!r}."
            )


#: Hookable tap points for a real `FastWAM` root module.
#:
#: **Statically derived from the class definitions at `ac7d376`; never executed
#: against a live model.** Attribute existence is pinned by
#: `test_candidate_fastwam_taps_resolve_on_a_structural_stub`, but firing order,
#: shapes and the right `feature_dim` are all unverified. The VAE latent is
#: deliberately absent: `WAMModeAdapter` is not an `nn.Module`, so that locus is
#: obtained by calling `encode_world_state()` and passing the returned
#: `first_frame_latents` straight to `probe_taps`, which accepts plain arrays.
CANDIDATE_FASTWAM_TAPS: tuple[TapSpec, ...] = (
    # `model.vae` is an nn.Module, so this is the hookable route to locus (a).
    # feature_dim=1 assumes a [B, C, T, H, W] latent -- confirm before use.
    TapSpec("vae_latent", "vae", feature_dim=1),
    # `video_expert.blocks` is an nn.ModuleList; index it for locus (b).
    TapSpec("video_block_0", "video_expert.blocks.0", feature_dim=-1),
    # Locus (c): the *input* to the action head is the readout state that
    # `post_dit` would have consumed.
    TapSpec("action_readout", "action_expert.head", site="input", feature_dim=-1),
)


def resolve_module(root: nn.Module, module_path: str) -> nn.Module:
    """Resolve a dotted attribute path to a submodule, failing loudly.

    Silently returning ``None`` for a typo is the single easiest way to produce a
    probe result that looks fine and measures nothing, so every failure here is
    an exception that names what was actually available.
    """
    if not isinstance(root, nn.Module):
        raise TypeError(f"root must be an nn.Module, got {type(root).__name__}.")
    if module_path in ("", "."):
        return root

    current: Any = root
    walked: list[str] = []
    for part in module_path.split("."):
        if part.isdigit() and isinstance(current, (nn.Sequential, nn.ModuleList)):
            index = int(part)
            if index >= len(current):
                raise IndexError(
                    f"module path {module_path!r}: index {index} is out of range at "
                    f"{'.'.join(walked) or '<root>'} (length {len(current)})."
                )
            current = current[index]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            available = sorted(
                name for name, _ in getattr(current, "named_children", lambda: [])()
            )
            where = ".".join(walked) or "<root>"
            raise AttributeError(
                f"module path {module_path!r}: {where} has no child {part!r}. "
                f"Available children: {available or '(none)'}."
            )
        walked.append(part)

    if not isinstance(current, nn.Module):
        raise TypeError(
            f"module path {module_path!r} resolved to {type(current).__name__}, "
            "which is not an nn.Module and cannot carry a forward hook."
        )
    return current


def _select_tensor(payload: Any, spec: TapSpec) -> torch.Tensor:
    """Pull the tensor named by `spec.select` out of a hook payload."""
    value = payload
    if spec.select is None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"tap {spec.name!r}: {spec.site} is a {type(value).__name__}, not a "
                "Tensor. Set TapSpec.select to an int (tuple index) or str (dict key)."
            )
        return value

    if isinstance(spec.select, int):
        if not isinstance(value, (tuple, list)):
            raise TypeError(
                f"tap {spec.name!r}: select={spec.select} needs a tuple/list "
                f"{spec.site}, got {type(value).__name__}."
            )
        if spec.select >= len(value):
            raise IndexError(
                f"tap {spec.name!r}: select={spec.select} is out of range for a "
                f"{spec.site} of length {len(value)}."
            )
        value = value[spec.select]
    else:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"tap {spec.name!r}: select={spec.select!r} needs a mapping "
                f"{spec.site}, got {type(value).__name__}."
            )
        if spec.select not in value:
            raise KeyError(
                f"tap {spec.name!r}: {spec.site} mapping has no key "
                f"{spec.select!r}. Available keys: {sorted(map(str, value))}."
            )
        value = value[spec.select]

    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"tap {spec.name!r}: select={spec.select!r} produced a "
            f"{type(value).__name__}, not a Tensor."
        )
    return value


@dataclass
class _Registered:
    spec: TapSpec
    handle: Any


class ActivationTaps:
    """Context manager that captures named activations and always cleans up.

    ``with`` guarantees `__exit__` runs, and `__exit__` removes every handle in a
    ``finally`` so one bad removal cannot strand the rest. A partially-registered
    `__enter__` also rolls itself back. Leaked hooks are worse than a crash: they
    silently slow and mutate every later forward pass in the process.

    Example::

        specs = [TapSpec("readout", "action_expert.blocks.11")]
        with ActivationTaps(model, specs) as taps:
            model(...)
        acts = taps.stack("readout")   # [N, ...]
    """

    def __init__(
        self,
        root: nn.Module,
        specs: Iterable[TapSpec],
        *,
        to_cpu: bool = True,
        detach: bool = True,
        dtype: torch.dtype | None = torch.float32,
        max_calls_per_tap: int | None = None,
    ) -> None:
        specs = list(specs)
        if not specs:
            raise ValueError("ActivationTaps needs at least one TapSpec.")
        duplicates = sorted(
            {s.name for s in specs if sum(o.name == s.name for o in specs) > 1}
        )
        if duplicates:
            raise ValueError(f"duplicate tap names: {duplicates}")

        self._root = root
        self._specs = {s.name: s for s in specs}
        self._to_cpu = bool(to_cpu)
        self._detach = bool(detach)
        self._dtype = dtype
        self._max_calls = max_calls_per_tap
        self._registered: list[_Registered] = []
        self._buffers: dict[str, list[torch.Tensor]] = {s.name: [] for s in specs}
        # Resolve every path up front: a typo must fail before any forward runs,
        # not produce an empty result an hour later.
        self._modules = {s.name: resolve_module(root, s.module_path) for s in specs}

    # -- capture ---------------------------------------------------------- #

    def _store(self, spec: TapSpec, payload: Any) -> None:
        buf = self._buffers[spec.name]
        if self._max_calls is not None and len(buf) >= self._max_calls:
            return
        tensor = _select_tensor(payload, spec)
        if self._detach:
            tensor = tensor.detach()
        if self._to_cpu:
            tensor = tensor.to("cpu")
        if self._dtype is not None:
            tensor = tensor.to(self._dtype)
        buf.append(tensor.clone())

    def _make_output_hook(self, spec: TapSpec):
        def hook(module, args, output):  # noqa: ANN001 - torch hook signature
            self._store(spec, output)

        return hook

    def _make_input_hook(self, spec: TapSpec):
        def hook(module, args):  # noqa: ANN001 - torch pre-hook signature
            # A pre-hook always receives the positional args tuple. When the
            # caller did not ask for a specific element, unwrap the common
            # single-argument case so `select=None` means "the input tensor".
            payload = args
            if spec.select is None and isinstance(args, tuple) and len(args) == 1:
                payload = args[0]
            self._store(spec, payload)

        return hook

    # -- context manager -------------------------------------------------- #

    def __enter__(self) -> "ActivationTaps":
        try:
            for name, spec in self._specs.items():
                module = self._modules[name]
                if spec.site == "output":
                    handle = module.register_forward_hook(self._make_output_hook(spec))
                else:
                    handle = module.register_forward_pre_hook(
                        self._make_input_hook(spec)
                    )
                self._registered.append(_Registered(spec, handle))
        except Exception:
            self._remove_all()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._remove_all()
        return False  # never swallow the caller's exception

    def _remove_all(self) -> None:
        errors: list[BaseException] = []
        for entry in self._registered:
            try:
                entry.handle.remove()
            except BaseException as err:  # pragma: no cover - defensive
                errors.append(err)
        self._registered.clear()
        if errors:
            raise RuntimeError(f"failed to remove {len(errors)} tap hook(s)") from errors[0]

    @property
    def active(self) -> bool:
        return bool(self._registered)

    # -- results ---------------------------------------------------------- #

    @property
    def names(self) -> list[str]:
        return sorted(self._specs)

    def spec(self, name: str) -> TapSpec:
        self._check_name(name)
        return self._specs[name]

    def _check_name(self, name: str) -> None:
        if name not in self._specs:
            raise KeyError(
                f"unknown tap {name!r}. Declared taps: {self.names}."
            )

    def get(self, name: str) -> list[torch.Tensor]:
        self._check_name(name)
        return list(self._buffers[name])

    def stack(self, name: str, dim: int = 0) -> torch.Tensor:
        """Concatenate everything captured for `name` along the batch axis."""
        self._check_name(name)
        chunks = self._buffers[name]
        if not chunks:
            raise RuntimeError(
                f"tap {name!r} captured nothing. The tapped module "
                f"({self._specs[name].module_path!r}) never ran during the "
                "`with` block."
            )
        return torch.cat(chunks, dim=dim)

    def counts(self) -> dict[str, int]:
        return {name: len(buf) for name, buf in self._buffers.items()}

    def clear(self) -> None:
        for buf in self._buffers.values():
            buf.clear()
