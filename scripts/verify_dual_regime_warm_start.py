"""GPU acceptance check for standalone-IDM -> shared S0 numerical parity.

This is intentionally a pre-training check. It strictly imports a standalone
IDM checkpoint into FusedDualRegimeFastWAM, then runs both IDM paths on the same
real dataset state with identical solver settings and random seed.
"""
from __future__ import annotations

import argparse
import gc
import os
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, help="resolved config.yaml from E-I")
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--target-task", required=True)
    parser.add_argument("--ckpt", required=True, help="standalone FastWAMIDM checkpoint")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=5e-3)
    args = parser.parse_args()

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from fastwam.adaptive_gate.warm_start import strict_standalone_idm_warm_start
    from fastwam.utils import misc

    if args.sample_index < 0 or args.inference_steps <= 0:
        parser.error("--sample-index must be non-negative and --inference-steps positive")
    stats_path = os.path.abspath(os.path.expanduser(args.dataset_stats))
    source_config_path = os.path.abspath(os.path.expanduser(args.source_config))
    checkpoint_path = os.path.abspath(os.path.expanduser(args.ckpt))
    for path in (stats_path, source_config_path, checkpoint_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    with initialize_config_dir(version_base="1.3", config_dir=configs_dir):
        target_cfg = compose(config_name="train", overrides=[f"task={args.target_task}"])
    source_cfg = OmegaConf.load(source_config_path)
    source_model_cfg = source_cfg.model if "model" in source_cfg else source_cfg

    with tempfile.TemporaryDirectory(prefix="fastwam-s0-parity-") as work_dir:
        misc.register_work_dir(work_dir)
        dataset = instantiate(
            target_cfg.data.train,
            pretrained_norm_stats=stats_path,
        )
        sample = dataset.get_strict(args.sample_index)

    input_image = sample["video"][:, 0].unsqueeze(0).to(args.device, dtype=dtype)
    proprio = sample.get("proprio")
    if proprio is not None:
        proprio = proprio[0].unsqueeze(0).to(args.device, dtype=dtype)
    context = sample["context"].unsqueeze(0).to(args.device, dtype=dtype)
    context_mask = sample["context_mask"].unsqueeze(0).to(args.device)
    action_horizon = int(sample["action"].shape[0])
    num_video_frames = (
        (int(dataset.num_frames) - 1) // int(dataset.action_video_freq_ratio) + 1
    )
    common = {
        "prompt": None,
        "input_image": input_image,
        "action_horizon": action_horizon,
        "num_video_frames": num_video_frames,
        "proprio": proprio,
        "context": context,
        "context_mask": context_mask,
        "num_inference_steps": args.inference_steps,
        "sigma_shift": args.sigma_shift,
        "seed": args.seed,
    }

    # Run sequentially so the acceptance check never holds two 5B models on
    # one GPU at once.
    source = instantiate(source_model_cfg, model_dtype=dtype, device=args.device)
    source.load_checkpoint(checkpoint_path)
    source.eval().requires_grad_(False)
    with torch.inference_mode():
        source_action = source.infer_action(**common)["action"].float().cpu()
    del source
    gc.collect()
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    target = instantiate(target_cfg.model, model_dtype=dtype, device=args.device)
    strict_standalone_idm_warm_start(
        target,
        {
            "kind": "standalone_idm",
            "checkpoint": checkpoint_path,
            "expected_checkpoint_sha256": args.checkpoint_sha256,
            "source_task": args.source_task,
            "source_config": source_config_path,
            "source_dataset_stats": stats_path,
        },
        target_model_config=target_cfg.model,
        target_dataset_stats=stats_path,
    )
    target.eval().requires_grad_(False)
    with torch.inference_mode():
        target_action = target.infer_action(
            **common, force_branch="idm", return_routing_info=True
        )["action"].float().cpu()

    torch.testing.assert_close(
        target_action,
        source_action,
        atol=args.atol,
        rtol=args.rtol,
        msg=(
            "S0 forced-IDM output differs from its standalone IDM parent under "
            "the paired seed/solver acceptance setting."
        ),
    )
    max_abs = float((target_action - source_action).abs().max().item())
    print(
        "PASS standalone->S0 forced-IDM parity: "
        f"sample={args.sample_index} seed={args.seed} max_abs={max_abs:.6g} "
        f"atol={args.atol} rtol={args.rtol}"
    )


if __name__ == "__main__":
    main()
