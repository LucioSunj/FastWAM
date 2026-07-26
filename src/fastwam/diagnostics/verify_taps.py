"""Real-model verification harness for the E7 three-point probe taps (W22).

`fastwam.diagnostics.taps.CANDIDATE_FASTWAM_TAPS` has only ever been checked
*statically*: attribute paths were pinned against a structural stub, but no live
FastWAM forward has ever run through the hooks. Two of the three tap points were
already wrong once on first writing (methods rather than submodules), so
decision point DP2's measurement foundation must be verified mechanically, not
by reading. This module is that mechanism.

`verify_taps(model)` registers every candidate tap through the W20
`ActivationTaps` context manager, runs exactly **one** forward, and reports per
tap: how many times it fired, the captured tensor's shape / dtype / device, and
the pooled feature width exactly as `fastwam.diagnostics.probe.pool_activation`
would compute it. The contract is fail-closed:

* a tap that resolves to nothing raises immediately from W20's
  `resolve_module` (that error is deliberately not caught here);
* a tap that resolves but **fires zero times** during the forward raises
  `TapVerificationError` naming the tap and its module path;
* a tap that fires more than once per forward is *reported* with its count
  (solver loops legitimately re-enter modules), never hidden.

A zero-fire outcome is a real and anticipated possibility, not a hypothetical:
statically, `FastWAM._encode_input_image_latents_tensor`
(`models/wan22/fastwam.py:382`) reaches the VAE via ``self.vae.encode(...)`` and
`WanVideoVAE38.encode` (`models/wan22/wan_video_vae.py:1218`) is a plain method
chain that never enters ``Module.__call__`` — so the ``vae_latent`` hook may
never fire on a real model even though the path resolves. Only a live run can
settle it; if it fails, that is a *finding* against `CANDIDATE_FASTWAM_TAPS`
(W20's contract) to report upstream, and the documented fallback is to call
`WAMModeAdapter.encode_world_state` and hand its `first_frame_latents` straight
to `probe_taps`, which accepts plain arrays.

One-command real-model verification, once a checkpoint and the Wan assets
exist (run from the FastWAM repo root)::

    python -m fastwam.diagnostics.verify_taps \
        --task libero_idm_2cam224_1e-4 \
        --ckpt /path/to/checkpoint.pt \
        --out runs/diagnostics/tap_verification.json \
        --device cuda

Assets are absent in the development environment, so that command is NOT-RUN
by definition here; ``--self-test`` proves the harness itself end-to-end
against a small built-in module tree that mimics the candidate tap structure::

    python -m fastwam.diagnostics.verify_taps --self-test

Heavy dependencies (hydra, the runtime factories, transformers via Wan loading)
are imported lazily inside the CLI's real-model path only, so this module —
like the rest of `fastwam.diagnostics` — imports with nothing but torch and
numpy available.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from .probe import DEFAULT_POOL_DIM, POOL_LAYOUT, pool_activation
from .taps import CANDIDATE_FASTWAM_TAPS, ActivationTaps, TapSpec

__all__ = [
    "TapFiring",
    "TapVerificationError",
    "TapVerificationReport",
    "build_synthetic_sample",
    "main",
    "verify_taps",
]


class TapVerificationError(RuntimeError):
    """A resolvable tap never fired (or could not be pooled) during the forward.

    Carries the partial ``report`` covering the taps that *did* fire, so a
    failure still tells the operator everything the run learned.
    """

    def __init__(self, message: str, report: "TapVerificationReport | None" = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class TapFiring:
    """What one tap actually did during the single verified forward."""

    name: str
    module_path: str
    site: str
    fired: int
    shape: tuple
    shape_varies: bool
    dtype: str
    device: str
    feature_dim: int
    pooled_feature_dim: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "module_path": self.module_path,
            "site": self.site,
            "fired": self.fired,
            "shape": list(self.shape),
            "shape_varies": self.shape_varies,
            "dtype": self.dtype,
            "device": self.device,
            "feature_dim": self.feature_dim,
            "pooled_feature_dim": self.pooled_feature_dim,
        }


@dataclass(frozen=True)
class TapVerificationReport:
    """Per-tap firing evidence from exactly one forward pass."""

    taps: Mapping[str, TapFiring]
    pool_output_dim: int
    pool_layout: str = POOL_LAYOUT

    def to_dict(self) -> dict:
        return {
            "pool_layout": self.pool_layout,
            "pool_output_dim": self.pool_output_dim,
            "taps": {name: firing.to_dict() for name, firing in sorted(self.taps.items())},
        }


# --------------------------------------------------------------------------- #
# Synthetic sample + default forward
# --------------------------------------------------------------------------- #

def build_synthetic_sample(
    model: Any,
    *,
    image_hw: tuple = (224, 448),
    context_len: int = 128,
    context_dim: int | None = None,
    action_horizon: int = 32,
    num_video_frames: int = 33,
    num_inference_steps: int = 2,
    seed: int = 0,
) -> dict:
    """Deterministic kwargs for a FastWAM-style ``infer_action`` call.

    Everything is zeros/ones (no RNG), so two calls with the same arguments are
    bitwise identical. Dimensions are read off the model where it exposes them
    (``proprio_dim``); the text-context width falls back to Wan2.2's 4096 when
    the model does not carry it. Kwargs the target signature does not accept are
    filtered out by `_call_with_supported_kwargs`, so this one builder serves
    every FastWAM variant.
    """
    height, width = int(image_hw[0]), int(image_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"image_hw must be positive, got {(height, width)}.")
    if context_dim is None:
        context_dim = getattr(
            getattr(model, "action_expert", None), "text_dim", None
        ) or getattr(getattr(model, "video_expert", None), "text_dim", None) or 4096

    sample: dict = {
        "prompt": None,
        "input_image": torch.zeros(1, 3, height, width),
        "action_horizon": int(action_horizon),
        "context": torch.zeros(1, int(context_len), int(context_dim)),
        "context_mask": torch.ones(1, int(context_len), dtype=torch.bool),
        "num_inference_steps": int(num_inference_steps),
        "num_video_frames": int(num_video_frames),
        "seed": int(seed),
        "rand_device": "cpu",
    }
    proprio_dim = getattr(model, "proprio_dim", None)
    if proprio_dim is not None:
        sample["proprio"] = torch.zeros(1, int(proprio_dim))
    return sample


def _call_with_supported_kwargs(func: Callable, kwargs: Mapping[str, Any]) -> Any:
    """Call ``func`` with only the kwargs its signature accepts.

    FastWAM variants differ in their ``infer_action`` signatures (e.g.
    ``num_video_frames`` / ``force_branch`` exist only on some); silently
    passing an unknown kwarg would raise, and dropping a *supported* one would
    change behaviour, so the filter is signature-driven rather than a hard-coded
    list. A ``**kwargs`` catch-all receives everything.
    """
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):  # builtins / C callables: pass through
        return func(**dict(kwargs))
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return func(**dict(kwargs))
    accepted = {k: v for k, v in kwargs.items() if k in parameters}
    return func(**accepted)


def _default_forward(model: Any, sample: Any) -> Any:
    """One forward through the model's action-inference path (or plain call).

    * A model exposing ``infer_action`` is driven through it, with a synthetic
      observation from `build_synthetic_sample` when no sample was supplied.
    * Any other module is called directly and requires an explicit ``sample``.
    """
    infer_action = getattr(model, "infer_action", None)
    if callable(infer_action):
        if sample is None:
            sample = build_synthetic_sample(model)
        if not isinstance(sample, Mapping):
            raise TypeError(
                "for a model with infer_action, `sample` must be a mapping of "
                f"infer_action kwargs, got {type(sample).__name__}. Use "
                "build_synthetic_sample() or pass forward_fn."
            )
        return _call_with_supported_kwargs(infer_action, sample)
    if sample is None:
        raise ValueError(
            "model has no infer_action and no `sample` was given; pass "
            "`sample` (tensor or kwargs mapping) or a `forward_fn`."
        )
    if isinstance(sample, Mapping):
        return model(**sample)
    return model(sample)


# --------------------------------------------------------------------------- #
# Verification core
# --------------------------------------------------------------------------- #

def verify_taps(
    model: nn.Module,
    taps: Sequence[TapSpec] = CANDIDATE_FASTWAM_TAPS,
    forward_fn: Callable[[Any, Any], Any] | None = None,
    sample: Any = None,
    *,
    pool_output_dim: int = DEFAULT_POOL_DIM,
) -> TapVerificationReport:
    """Register ``taps``, run exactly one forward, and report what fired.

    Path-resolution failures (typo, method-not-module, out-of-range index)
    propagate unchanged from W20's fail-closed `ActivationTaps` constructor.
    After the forward, any tap with zero captures raises `TapVerificationError`
    naming every silent tap and its module path; the exception carries the
    partial report for the taps that did fire. Multiple fires per forward are
    normal for solver loops and are reported via ``fired``, never suppressed.
    """
    specs = list(taps)
    # dtype=None / to_cpu=False: report the dtype and device the activation
    # really had. Converting to float32-on-CPU first (the ActivationTaps
    # default, right for probing) would misreport both.
    with ActivationTaps(model, specs, to_cpu=False, detach=True, dtype=None) as handle:
        runner = forward_fn if forward_fn is not None else _default_forward
        runner(model, sample)

    counts = handle.counts()
    firings: dict[str, TapFiring] = {}
    silent: list[TapSpec] = []
    for spec in specs:
        fired = counts[spec.name]
        if fired == 0:
            silent.append(spec)
            continue
        captured = handle.get(spec.name)
        first = captured[0]
        try:
            pooled = pool_activation(
                first, feature_dim=spec.feature_dim, output_dim=pool_output_dim
            )
        except Exception as err:
            raise TapVerificationError(
                f"tap {spec.name!r} (module_path={spec.module_path!r}) fired but "
                f"its activation could not be pooled with "
                f"feature_dim={spec.feature_dim}: {err}",
                report=TapVerificationReport(dict(firings), int(pool_output_dim)),
            ) from err
        firings[spec.name] = TapFiring(
            name=spec.name,
            module_path=spec.module_path,
            site=spec.site,
            fired=fired,
            shape=tuple(first.shape),
            shape_varies=any(t.shape != first.shape for t in captured[1:]),
            dtype=str(first.dtype),
            device=str(first.device),
            feature_dim=int(spec.feature_dim),
            pooled_feature_dim=int(pooled.shape[-1]),
        )

    if silent:
        report = TapVerificationReport(dict(firings), int(pool_output_dim))
        described = "; ".join(
            f"{spec.name} (module_path={spec.module_path!r}, site={spec.site!r})"
            for spec in silent
        )
        raise TapVerificationError(
            f"tap(s) resolved but never fired during the forward: {described}. "
            f"The module exists but the executed path does not call it via "
            f"Module.__call__ (a plain method call, e.g. vae.encode(...), does "
            f"not trigger forward hooks). Fired counts: {counts}.",
            report=report,
        )
    return TapVerificationReport(dict(firings), int(pool_output_dim))


# --------------------------------------------------------------------------- #
# Built-in self-test model (no assets, no heavy deps)
# --------------------------------------------------------------------------- #

class _SelfTestVAE(nn.Module):
    """Stands in for `model.vae`; emits a [B, C, T, H, W] latent via forward."""

    def __init__(self, channels: int = 6):
        super().__init__()
        self.lift = nn.Conv2d(3, channels, kernel_size=1)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        return self.lift(pooled).unsqueeze(2)  # [B, C, 1, 4, 4]


class _SelfTestBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x))


class _SelfTestVideoExpert(nn.Module):
    def __init__(self, dim: int, n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_SelfTestBlock(dim) for _ in range(n_blocks)])


class _SelfTestActionExpert(nn.Module):
    def __init__(self, dim: int, action_dim: int = 7):
        super().__init__()
        self.head = nn.Linear(dim, action_dim)


class _SelfTestFastWAM(nn.Module):
    """Minimal module tree with the candidate tap paths *and* an infer_action.

    Mirrors the structure `CANDIDATE_FASTWAM_TAPS` names — ``vae``,
    ``video_expert.blocks.N``, ``action_expert.head`` — and drives all three
    through a FastWAM-shaped ``infer_action``, so ``--self-test`` exercises the
    default-forward path (synthetic sample construction, signature filtering,
    tap capture, pooling) end to end without any real assets.
    """

    proprio_dim = 8

    def __init__(self, dim: int = 12):
        super().__init__()
        self.vae = _SelfTestVAE()
        self.video_expert = _SelfTestVideoExpert(dim)
        self.action_expert = _SelfTestActionExpert(dim)
        self.to_tokens = nn.Linear(self.vae.channels, dim)

    @torch.no_grad()
    def infer_action(
        self,
        prompt=None,
        input_image: torch.Tensor | None = None,
        action_horizon: int = 32,
        proprio: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        num_inference_steps: int = 2,
        seed: int | None = None,
        **_: Any,
    ) -> dict:
        if input_image is None:
            raise ValueError("input_image is required.")
        latent = self.vae(input_image)                       # [B, C, 1, 4, 4]
        tokens = self.to_tokens(latent.flatten(2).transpose(1, 2))  # [B, 16, dim]
        for _step in range(max(1, int(num_inference_steps))):
            for block in self.video_expert.blocks:
                tokens = block(tokens)
        action = self.action_expert.head(tokens)
        return {"action": action[:, : int(action_horizon)]}

    def forward(self, x: torch.Tensor) -> dict:
        return self.infer_action(input_image=x)


def _run_self_test(pool_output_dim: int) -> dict:
    torch.manual_seed(0)
    model = _SelfTestFastWAM().eval()
    report = verify_taps(
        model,
        CANDIDATE_FASTWAM_TAPS,
        sample=build_synthetic_sample(model, image_hw=(16, 32), num_inference_steps=1),
        pool_output_dim=pool_output_dim,
    )
    payload = report.to_dict()
    payload["self_test"] = True
    return payload


# --------------------------------------------------------------------------- #
# CLI (real-model path imports everything heavy lazily)
# --------------------------------------------------------------------------- #

def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remediation(err: BaseException) -> str:
    """One line naming what is missing and how to get it."""
    if isinstance(err, ModuleNotFoundError):
        return (
            f"remediation: python package {err.name!r} is not installed - "
            "pip install it (version pinned in FastWAM pyproject.toml)."
        )
    if isinstance(err, FileNotFoundError):
        return (
            "remediation: a required asset is missing - check the path, and for "
            "Wan2.2 weights set DIFFSYNTH_MODEL_BASE_PATH to a local checkout "
            "(see FastWAM README, External Assets Required)."
        )
    return (
        "remediation: model construction failed - verify the task config, the "
        "checkpoint/task match, and that Wan2.2 assets plus the preprocessed "
        "ActionDiT backbone are available (FastWAM README)."
    )


def _construct_real_model(args: argparse.Namespace):
    """Compose the task config and build the model via the runtime factories.

    Imported lazily so `import fastwam.diagnostics.verify_taps` works in an
    environment without hydra / transformers / accelerate / Wan assets.
    """
    try:
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
    except ModuleNotFoundError as err:
        print(_remediation(err), file=sys.stderr)
        raise

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "error: --device cuda requested but torch.cuda.is_available() is "
            "False. remediation: run on a GPU machine or pass --device cpu."
        )

    task_path = Path(args.task)
    if task_path.suffix in (".yaml", ".yml") and task_path.exists():
        cfg = OmegaConf.load(task_path)
        if "model" not in cfg:
            raise SystemExit(
                f"error: {task_path} has no `model` section; pass a fully "
                "composed config dump or a task name from configs/task/."
            )
    else:
        config_dir = Path(__file__).resolve().parents[3] / "configs"
        if not (config_dir / "train.yaml").exists():
            raise SystemExit(
                f"error: cannot locate configs/train.yaml under {config_dir}; "
                "run from a FastWAM checkout or pass --task as a config path."
            )
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
            cfg = compose(config_name="train", overrides=[f"task={args.task}"])

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    try:
        model = instantiate(cfg.model, model_dtype=dtype, device=args.device)
    except Exception as err:
        print(_remediation(err), file=sys.stderr)
        raise
    try:
        model.load_checkpoint(str(args.ckpt))
    except Exception as err:
        print(
            "remediation: checkpoint could not be loaded - confirm it matches "
            f"the task config ({args.task}) and was produced by FastWAM "
            "training (weights payload with `mot`).",
            file=sys.stderr,
        )
        raise err
    model = model.to(args.device).eval()

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    digest = hashlib.sha256(
        json.dumps(model_cfg, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    image_hw = args.image_hw
    if image_hw is None:
        video_size = OmegaConf.select(cfg, "data.train.video_size", default=None)
        image_hw = tuple(video_size) if video_size is not None else (224, 448)
    context_len = args.context_len
    if context_len is None:
        context_len = int(OmegaConf.select(cfg, "model.tokenizer_max_len", default=128))
    context_dim = int(
        OmegaConf.select(cfg, "model.action_dit_config.text_dim", default=None)
        or OmegaConf.select(cfg, "model.video_dit_config.text_dim", default=None)
        or 4096
    )
    num_video_frames = int(OmegaConf.select(cfg, "data.train.num_frames", default=33))
    return model, digest, tuple(int(v) for v in image_hw), context_len, context_dim, num_video_frames


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fastwam.diagnostics.verify_taps",
        description=(
            "Verify mechanically that CANDIDATE_FASTWAM_TAPS fire on a live "
            "model: one forward, per-tap fired counts / shapes / dtypes / "
            "pooled feature widths, fail-closed on any silent tap."
        ),
    )
    parser.add_argument("--task", default=None,
                        help="hydra task name (configs/task/) or a composed config YAML path")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="FastWAM checkpoint (.pt) to load before verifying")
    parser.add_argument("--dataset-stats", type=Path, default=None,
                        help="optional dataset_stats.json; recorded by hash only")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the JSON report here (required unless --self-test)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--image-hw", type=int, nargs=2, metavar=("H", "W"), default=None,
                        help="synthetic image size; defaults to the task's video_size")
    parser.add_argument("--context-len", type=int, default=None,
                        help="synthetic text-context length; defaults to tokenizer_max_len")
    parser.add_argument("--steps", type=int, default=2,
                        help="num_inference_steps for the single verification forward")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-output-dim", type=int, default=DEFAULT_POOL_DIM)
    parser.add_argument("--self-test", action="store_true",
                        help="verify the harness against a built-in fake module tree")
    args = parser.parse_args(argv)

    if args.self_test:
        payload = _run_self_test(args.pool_output_dim)
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
        print(text)
        return 0

    if args.task is None or args.ckpt is None:
        parser.error("--task and --ckpt are required (or pass --self-test)")
    if args.out is None:
        parser.error("--out is required for a real-model run")
    if not args.ckpt.exists():
        parser.error(f"--ckpt {args.ckpt} does not exist")
    if args.dataset_stats is not None and not args.dataset_stats.exists():
        parser.error(f"--dataset-stats {args.dataset_stats} does not exist")

    (
        model,
        config_digest,
        image_hw,
        context_len,
        context_dim,
        num_video_frames,
    ) = _construct_real_model(args)

    sample = build_synthetic_sample(
        model,
        image_hw=image_hw,
        context_len=context_len,
        context_dim=context_dim,
        action_horizon=args.action_horizon,
        num_video_frames=num_video_frames,
        num_inference_steps=args.steps,
        seed=args.seed,
    )
    try:
        with torch.no_grad():
            report = verify_taps(
                model, CANDIDATE_FASTWAM_TAPS, sample=sample,
                pool_output_dim=args.pool_output_dim,
            )
    except TapVerificationError as err:
        # Fail-closed: no report file that could be mistaken for a pass.
        print(f"TAP VERIFICATION FAILED: {err}", file=sys.stderr)
        if err.report is not None and err.report.taps:
            print(
                "taps that did fire: "
                + json.dumps(err.report.to_dict()["taps"], sort_keys=True),
                file=sys.stderr,
            )
        return 3

    payload = report.to_dict()
    payload.update(
        {
            "task": str(args.task),
            "device": args.device,
            "dtype": args.dtype,
            "image_hw": [int(image_hw[0]), int(image_hw[1])],
            "context_len": int(context_len),
            "num_inference_steps": int(args.steps),
            "action_horizon": int(args.action_horizon),
            "seed": int(args.seed),
            "model_config_digest": config_digest,
            "ckpt_sha256": _sha256_of_file(args.ckpt),
            "dataset_stats_sha256": (
                _sha256_of_file(args.dataset_stats)
                if args.dataset_stats is not None
                else None
            ),
        }
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
