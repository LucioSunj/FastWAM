"""Profile future-content controls without changing the production cost table.

The synthetic-input timings here calibrate solver work and verify that NoRead
and ExtraCompute are approximately compute matched. Closed-loop experiment
latency remains the headline end-to-end measurement.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from profile_wam_modes import (
    _build_model,
    _measure_flops,
    _measure_latency_ms,
    _synthetic_obs,
)


def _device_name(device: str) -> str:
    if not device.startswith("cuda"):
        return device.split(":", 1)[0]
    import torch

    return torch.cuda.get_device_name(torch.device(device))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="libero_dual_regime_fused_2cam224_1e-4")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset-stats-sha256", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    ap.add_argument("--inference-steps", type=int, default=20)
    ap.add_argument("--sigma-shift", type=float, default=None)
    ap.add_argument("--max-extra-action-steps", type=int, default=80)
    ap.add_argument("--latency-match-tolerance", type=float, default=0.05)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--num-video-frames", type=int, default=9)
    ap.add_argument("--action-horizon", type=int, default=32)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--latency-iters", type=int, default=5)
    ap.add_argument("--allow-legacy-checkpoint", action="store_true")
    ap.add_argument("--allow-unmatched", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.latency_iters < 1 or args.inference_steps < 1:
        ap.error("inference and latency iteration counts must be positive")
    if args.max_extra_action_steps < args.inference_steps:
        ap.error("--max-extra-action-steps must be >= --inference-steps")
    if not 0.0 <= args.latency_match_tolerance < 1.0:
        ap.error("--latency-match-tolerance must be in [0,1)")

    import yaml

    from fastwam.adaptive_gate import IDMControl, ShuffledFutureDonor, WAMModeAdapter

    model = _build_model(args.task, args.device, args.ckpt, args.dtype)
    adapter = WAMModeAdapter(
        model,
        backbone_kind="idm",
        task=args.task,
        num_video_frames=args.num_video_frames,
        generation_horizon=args.action_horizon,
        inference_steps=args.inference_steps,
        sigma_shift=args.sigma_shift,
        context_len=args.context_len,
        default_seed=0,
        dataset_stats_fingerprint=args.dataset_stats_sha256,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )
    obs, _ = _synthetic_obs(
        model, args.height, args.width, args.action_horizon, args.context_len, args.device
    )
    encoded = adapter.encode_world_state(obs["input_image"])
    donor_out = adapter.act_control(
        input_image=obs["input_image"],
        control=IDMControl.VALID_IDM,
        proprio=obs["proprio"],
        context=obs["context"],
        context_mask=obs["context_mask"],
        encoded_state=encoded,
        seed=1,
        return_video_latents=True,
    )
    donor_meta = {
        "task": args.task,
        "factor": "profile",
        "level": 0,
        "phase": "profile",
        "ckpt_fingerprint": model._loaded_checkpoint_fingerprint,
        "solver_steps": args.inference_steps,
        "solver_fingerprint": adapter.solver_fingerprint,
    }
    donor = ShuffledFutureDonor(donor_out["video_latents"], donor_meta)

    def call_control(control: IDMControl, *, extra_steps: int | None = None):
        def _call():
            state = adapter.encode_world_state(obs["input_image"])
            adapter.act_control(
                input_image=obs["input_image"],
                control=control,
                proprio=obs["proprio"],
                context=obs["context"],
                context_mask=obs["context_mask"],
                encoded_state=state,
                seed=0,
                shuffled_future_donor=(donor if control is IDMControl.SHUFFLED else None),
                expected_donor_metadata=(donor_meta if control is IDMControl.SHUFFLED else None),
                extra_action_steps=extra_steps,
            )

        return _call

    raw: dict[str, dict[str, float | int | bool]] = {}
    for control in (
        IDMControl.VALID_IDM,
        IDMControl.NO_READ,
        IDMControl.REPEAT_CURRENT,
        IDMControl.SHUFFLED,
    ):
        fn = call_control(control)
        fn()
        raw[control.value] = {
            "flops": _measure_flops(fn),
            "latency_ms": _measure_latency_ms(fn, args.latency_iters, args.device),
            "action_steps": args.inference_steps,
        }
        print(f"[{control.value}] {raw[control.value]}")

    idm_latency = float(raw[IDMControl.VALID_IDM.value]["latency_ms"])
    best: tuple[float, int, float] | None = None
    # Search integer action-step counts. FLOPs are measured only for the winner.
    for steps in range(args.inference_steps, args.max_extra_action_steps + 1):
        latency = _measure_latency_ms(
            call_control(IDMControl.EXTRA_COMPUTE, extra_steps=steps),
            args.latency_iters,
            args.device,
        )
        relative_error = abs(latency - idm_latency) / max(idm_latency, 1e-9)
        candidate = (relative_error, steps, latency)
        if best is None or candidate < best:
            best = candidate
        if relative_error <= args.latency_match_tolerance:
            break
    assert best is not None
    relative_error, extra_steps, extra_latency = best
    extra_fn = call_control(IDMControl.EXTRA_COMPUTE, extra_steps=extra_steps)
    raw[IDMControl.EXTRA_COMPUTE.value] = {
        "flops": _measure_flops(extra_fn),
        "latency_ms": extra_latency,
        "action_steps": extra_steps,
        "latency_relative_error_to_idm": relative_error,
        "compute_matched": relative_error <= args.latency_match_tolerance,
    }

    no_read_error = abs(float(raw["no_read"]["latency_ms"]) - idm_latency) / max(
        idm_latency, 1e-9
    )
    raw["no_read"]["latency_relative_error_to_idm"] = no_read_error
    raw["no_read"]["compute_matched"] = no_read_error <= args.latency_match_tolerance
    unmatched = [
        name for name in ("no_read", "extra_compute") if not bool(raw[name]["compute_matched"])
    ]
    if unmatched and not args.allow_unmatched:
        raise RuntimeError(
            f"Controls are not within {args.latency_match_tolerance:.1%} of IDM: {unmatched}."
        )

    payload = {
        "schema_version": 1,
        "kind": "fastwam_control_profile",
        "controls": raw,
        "meta": {
            "task": args.task,
            "ckpt_fingerprint": model._loaded_checkpoint_fingerprint,
            "ckpt_path": os.path.abspath(args.ckpt),
            "dataset_stats_fingerprint": args.dataset_stats_sha256,
            "inference_steps": args.inference_steps,
            "solver_contract": adapter.solver_contract,
            "solver_fingerprint": adapter.solver_fingerprint,
            "height": args.height,
            "width": args.width,
            "num_video_frames": args.num_video_frames,
            "action_horizon": args.action_horizon,
            "context_len": args.context_len,
            "model_dtype": str(model.torch_dtype),
            "device": args.device,
            "device_name": _device_name(args.device),
            "latency_match_tolerance": args.latency_match_tolerance,
        },
    }
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
