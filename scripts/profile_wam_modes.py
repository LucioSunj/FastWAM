"""One-time profiler: measure per-mode compute of WAMModeAdapter and write a cost table.

Measures both modes {UNCOND, IDM}:
  - FLOPs   via torch.utils.flop_counter.FlopCounterMode (hardware-independent)
  - latency via CUDA events (hardware-specific, for reference)
normalizes both so IDM == 1, and writes a cost YAML consumed by the reward
(`-lambda * cost(mode)`). The reward defaults to the FLOPs-normalized values.

HEAVY: needs the Wan2.2 weights + a GPU. Run once per checkpoint/deployment shape.

Example:
  cd FastWAM
  python scripts/profile_wam_modes.py \
    --task libero_dual_regime_fused_2cam224_1e-4 \
    --backbone-kind idm --ckpt /path/to/checkpoint.pt --inference-steps 20 \
    --height 224 --width 448 --num-video-frames 9 --action-horizon 32 \
    --out configs/adaptive_gate/wam_cost_libero_idm.yaml
"""
from __future__ import annotations

import argparse
import os


def _build_model(task: str, device: str, ckpt: str | None, dtype: str):
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    with initialize_config_dir(version_base="1.3", config_dir=configs_dir):
        cfg = compose(config_name="train", overrides=[f"task={task}"])
    model_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[str(dtype)]
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    if ckpt:
        model.load_checkpoint(ckpt)
    model.eval()
    model.requires_grad_(False)
    return model


def _synthetic_obs(model, height: int, width: int, action_horizon: int, context_len: int, device: str):
    import torch

    a_dim = int(model.action_expert.action_dim)
    proprio_dim = getattr(model, "proprio_dim", None)
    text_dim = int(model.action_expert.text_dim)
    ctx_len = int(context_len)
    obs = {
        "input_image": torch.rand(1, 3, height, width, device=device),
        "context": torch.randn(1, ctx_len, text_dim, device=device),
        "context_mask": torch.ones(1, ctx_len, dtype=torch.bool, device=device),
        "proprio": None if proprio_dim is None else torch.randn(1, int(proprio_dim), device=device),
    }
    return obs, a_dim


def _measure_flops(call_fn) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with counter:
        call_fn()
    return float(counter.get_total_flops())


def _measure_latency_ms(call_fn, iters: int, device: str) -> float:
    import torch

    if device.startswith("cuda"):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            call_fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / max(iters, 1))
    import time

    t0 = time.perf_counter()
    for _ in range(iters):
        call_fn()
    return float((time.perf_counter() - t0) * 1000.0 / max(iters, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="libero_dual_regime_fused_2cam224_1e-4")
    ap.add_argument("--backbone-kind", choices=["idm"], default="idm")
    ap.add_argument("--ckpt", required=True, help="dual-regime checkpoint to fingerprint")
    ap.add_argument("--allow-legacy-checkpoint", action="store_true",
                    help="allow a pre-provenance checkpoint after manual verification")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="must match actor.model.wam.dtype",
    )
    ap.add_argument("--inference-steps", type=int, default=20)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--num-video-frames", type=int, default=9)
    ap.add_argument("--action-horizon", type=int, default=32)
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--latency-iters", type=int, default=5)
    ap.add_argument("--out", default="configs/adaptive_gate/wam_cost.yaml")
    args = ap.parse_args()
    if args.latency_iters < 1:
        ap.error("--latency-iters must be >= 1")

    from fastwam.adaptive_gate import MODE_ORDER, WAMModeAdapter, normalize_cost_table, save_cost_table

    model = _build_model(args.task, args.device, args.ckpt, args.dtype)
    adapter = WAMModeAdapter(
        model,
        backbone_kind=args.backbone_kind,
        task=args.task,
        num_video_frames=args.num_video_frames,
        generation_horizon=args.action_horizon,
        inference_steps=args.inference_steps,
        context_len=args.context_len,
        default_seed=0,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )
    obs, _ = _synthetic_obs(
        model, args.height, args.width, args.action_horizon, args.context_len, args.device
    )
    def call_for(mode):
        def _fn():
            encoded_state = adapter.encode_world_state(obs["input_image"])
            adapter.act(
                input_image=obs["input_image"],
                mode=mode,
                proprio=obs["proprio"],
                context=obs["context"],
                context_mask=obs["context_mask"],
                encoded_state=encoded_state,
                seed=0,
            )
        return _fn

    flops, latency = {}, {}
    for mode in MODE_ORDER:
        fn = call_for(mode)
        fn()  # warmup
        flops[mode.value] = _measure_flops(fn)
        latency[mode.value] = _measure_latency_ms(fn, args.latency_iters, args.device)
        print(f"[{mode.value:6s}] flops={flops[mode.value]:.3e}  latency={latency[mode.value]:.1f} ms")

    normalized = normalize_cost_table(flops)
    if args.device.startswith("cuda"):
        import torch

        device_name = torch.cuda.get_device_name(torch.device(args.device))
    else:
        device_name = str(args.device).split(":", 1)[0]
    save_cost_table(
        args.out,
        normalized=normalized,
        source="flops",
        raw={"flops": flops, "latency_ms": latency},
        meta={
            "task": args.task,
            "backbone_kind": args.backbone_kind,
            "ckpt_fingerprint": model._loaded_checkpoint_fingerprint,
            "inference_steps": args.inference_steps,
            "height": args.height,
            "width": args.width,
            "num_video_frames": args.num_video_frames,
            "action_horizon": args.action_horizon,
            "context_len": args.context_len,
            "model_dtype": str(model.torch_dtype),
            "proprio_dim": getattr(model, "proprio_dim", None),
            "device": args.device,
            "device_name": device_name,
        },
    )
    print(f"\nnormalized cost (IDM=1): {normalized}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
