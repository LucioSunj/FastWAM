"""W22 vae_latent fallback check (Phase B1, TEST_PLAN section 3.1).

The ``vae_latent`` tap in ``CANDIDATE_FASTWAM_TAPS`` hooks the ``model.vae``
submodule via ``Module.__call__``, but ``FastWAM._encode_input_image_latents_tensor``
(``models/wan22/fastwam.py:381``) reaches the VAE through the plain method
chain ``self.vae.encode(...)`` (``wan_video_vae.py:1218``), which never enters
``Module.__call__`` -- so the hook cannot fire.  The real-hardware run
confirmed this pre-registered outcome (fired=0, harness rc=3, fail-closed).

The documented fallback (``fastwam/diagnostics/taps.py`` and
``fastwam/diagnostics/verify_taps.py`` module docstrings) is: obtain the
first-frame VAE latent by calling ``WAMModeAdapter.encode_world_state`` and
hand its ``first_frame_latents`` straight to
``fastwam.diagnostics.probe.probe_taps``, which accepts plain arrays.

This script:

1. builds the same model/checkpoint as the real ``verify_taps`` run, through
   the harness's own ``_construct_real_model`` (identical config composition
   and ``load_checkpoint`` path);
2. attempts the adapter route first.  ``WAMModeAdapter`` is expected to
   fail-close on this plain (non-dual-regime) FastWAMIDM checkpoint; the
   verbatim constructor error is recorded, and the script then performs the
   exact call ``encode_world_state`` makes internally
   (``model._encode_input_image_latents_tensor``, see
   ``adaptive_gate/wam_mode_adapter.py:325-348``), reproducing the
   ``EncodedWorldState`` shape and world-feature contract checks verbatim;
3. feeds the latent(s) as the VAE-layer activation into
   ``fastwam.diagnostics.probe`` end-to-end: ``pool_activation`` with the tap's
   declared ``feature_dim=1`` and then ``probe_taps`` (grouped cross-fitted
   linear probe).  The probe needs N >= n_splits samples with both classes, so
   N deterministic seeded synthetic inputs are encoded; sample 0 is the fixed
   input.  The AUC on random labels is meaningless by construction -- the
   acceptance criterion is that shapes/dtypes are accepted end-to-end without
   error.

New diagnostic file under ``runs/diagnostics/``; it does not modify ``src/``
or any existing test.  Exit code 0 iff the fallback path runs clean.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fastwam.diagnostics.probe import DEFAULT_POOL_DIM, pool_activation, probe_taps
from fastwam.diagnostics.verify_taps import _construct_real_model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    ns = argparse.Namespace(
        task=args.task,
        ckpt=args.ckpt,
        device=args.device,
        dtype=args.dtype,
        image_hw=None,
        context_len=None,
    )
    (
        model,
        config_digest,
        image_hw,
        context_len,
        context_dim,
        num_video_frames,
    ) = _construct_real_model(ns)

    result: dict = {
        "check": "w22_vae_latent_fallback_v1",
        "task": args.task,
        "ckpt": str(args.ckpt),
        "device": args.device,
        "dtype": args.dtype,
        "image_hw": [int(image_hw[0]), int(image_hw[1])],
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "model_config_digest": config_digest,
    }

    height, width = image_hw

    def synthetic_image(seed: int) -> torch.Tensor:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        return torch.rand((1, 3, height, width), generator=gen) * 2.0 - 1.0

    fixed_image = synthetic_image(args.seed)

    # ---- Route 1: the documented adapter entry point ---------------------- #
    latent = None
    encode_fn = None
    try:
        from fastwam.adaptive_gate.wam_mode_adapter import WAMModeAdapter

        adapter = WAMModeAdapter(
            model,
            backbone_kind="idm",
            task=args.task,
            num_video_frames=num_video_frames,
            generation_horizon=32,
        )
        state = adapter.encode_world_state(fixed_image)
        latent = state.first_frame_latents
        result["route"] = "WAMModeAdapter.encode_world_state"
        result["world_feat_dim"] = int(state.world_feat.numel())
        result["world_feat_dim_expected"] = int(adapter.world_feat_dim)
        encode_fn = lambda img: adapter.encode_world_state(img).first_frame_latents
    except Exception as err:  # expected fail-closed on a plain IDM checkpoint
        result["adapter_error"] = f"{type(err).__name__}: {err}"

    # ---- Route 2: the identical underlying call --------------------------- #
    if latent is None:
        @torch.no_grad()
        def encode_core(img: torch.Tensor) -> torch.Tensor:
            # Verbatim core of encode_world_state (wam_mode_adapter.py:325-333):
            dtype = getattr(model, "torch_dtype", img.dtype)
            device = getattr(model, "device", img.device)
            image = img.to(device=device, dtype=dtype)
            z = model._encode_input_image_latents_tensor(image)
            if z.ndim != 5 or z.shape[0] != 1:
                raise ValueError(
                    "first-frame VAE latent must be [1,C,T,H,W], got "
                    f"{tuple(z.shape)}."
                )
            return z.detach()

        latent = encode_core(fixed_image)
        encode_fn = encode_core
        result["route"] = (
            "model._encode_input_image_latents_tensor "
            "(the exact call encode_world_state performs internally)"
        )
        # Reproduce the EncodedWorldState world-feature contract check verbatim
        # (wam_mode_adapter.py:334-343).
        coarse = F.adaptive_avg_pool3d(latent.float(), (1, 2, 2)).flatten(1)
        spread = latent.float().std(dim=(2, 3, 4), unbiased=False)
        world_feat = torch.cat([coarse, spread], dim=-1).squeeze(0)
        expected = 5 * int(model.vae.model.z_dim)
        if world_feat.numel() != expected:
            raise RuntimeError(
                f"world feature dim mismatch: expected {expected}, got "
                f"{world_feat.numel()}."
            )
        result["world_feat_dim"] = int(world_feat.numel())
        result["world_feat_dim_expected"] = int(expected)

    result["first_frame_latents"] = {
        "shape": list(latent.shape),
        "dtype": str(latent.dtype),
        "device": str(latent.device),
        "vae_z_dim": int(model.vae.model.z_dim),
        "finite": bool(torch.isfinite(latent.float()).all()),
    }

    # ---- Feed the latent into fastwam.diagnostics.probe end-to-end -------- #
    n = int(args.n_samples)
    latents = [latent.float().cpu()]
    for i in range(1, n):
        latents.append(encode_fn(synthetic_image(args.seed + i)).float().cpu())
    activations = torch.cat(latents, dim=0)  # [N, C, T, H, W]
    result["stacked_activation_shape"] = list(activations.shape)

    # feature_dim=1 is exactly what TapSpec("vae_latent", "vae", feature_dim=1)
    # declares for a [B, C, T, H, W] latent.
    pooled = pool_activation(activations, feature_dim=1, output_dim=DEFAULT_POOL_DIM)
    result["pooled_shape"] = list(pooled.shape)
    result["pooled_feature_dim"] = int(pooled.shape[-1])

    labels = (np.arange(n) % 2).astype(np.int64)  # synthetic: AUC is meaningless
    groups = np.arange(n)
    probe_out = probe_taps(
        {"vae_latent": activations},
        labels,
        groups,
        feature_dims={"vae_latent": 1},
        output_dim=DEFAULT_POOL_DIM,
        n_splits=5,
        n_bootstrap=100,
        seed=args.seed,
    )
    probe_result = dataclasses.asdict(probe_out["vae_latent"])
    result["probe_result"] = probe_result
    result["probe_labels_note"] = (
        "labels are synthetic (alternating); the probe AUC is meaningless by "
        "construction. The acceptance criterion is only that the latent is "
        "accepted end-to-end by pool_activation + probe_taps."
    )
    result["end_to_end_ok"] = True

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
