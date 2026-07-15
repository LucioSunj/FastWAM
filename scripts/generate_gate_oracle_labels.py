"""Generate self-supervised ORACLE mode labels for the adaptive-prediction gate (M3).

For each sampled state, run the frozen dual-regime WAM in {UNCOND, IDM} with
paired action-noise seeds, measure action error, and label the CHEAPEST NEAR-BEST
mode. Absolute IDM quality/regret are retained so failed references stay visible.
the targets come from data the VLA dataset already contains.

The output shards feed the gate's BC warm-start (SFT) in RLinf:
    RLinf/examples/embodiment/train_gate_bc.py --labels 'labels/*.pt' ...

HEAVY: each sample runs both complete UNCOND and IDM action pipelines for every seed.
Subsample with --stride/--max-samples and
shard across GPUs with --num-shards/--shard-index (disjoint strided slices).

Example (LIBERO, IDM backbone, 4 GPUs -> 4 shards):
  cd FastWAM
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/generate_gate_oracle_labels.py \
      --task libero_dual_regime_fused_2cam224_1e-4 --backbone-kind idm \
      --ckpt /path/to/dual_regime_idm.pt \
      --dataset-stats /path/to/dataset_stats.json \
      --stride 20 --num-shards 4 --shard-index $i \
      --exec-horizon 10 --out data/gate_oracle/libero_idm &
  done; wait

Tolerances default to tol-rel=0.1, tol-abs=0.02 (normalized action units), but the
stored per-step error curves allow OFFLINE relabeling with different knobs via
`fastwam.adaptive_gate.relabel_from_steps` — no WAM re-run.
"""
from __future__ import annotations

import argparse
import hashlib
import os


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_model_and_dataset(args):
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    with initialize_config_dir(version_base="1.3", config_dir=configs_dir):
        cfg = compose(config_name="train", overrides=[f"task={args.task}"])

    resolved_stats = args.dataset_stats
    if resolved_stats is None:
        resolved_stats = cfg.data.train.get("pretrained_norm_stats", None)
    if not resolved_stats or not os.path.isfile(os.path.expanduser(str(resolved_stats))):
        raise FileNotFoundError(
            "Oracle generation requires the exact dataset_stats.json used by the "
            "checkpoint. Pass --dataset-stats or configure data.train.pretrained_norm_stats."
        )
    resolved_stats = os.path.abspath(os.path.expanduser(str(resolved_stats)))

    model_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[str(args.dtype)]
    model = instantiate(cfg.model, model_dtype=model_dtype, device=args.device)
    if args.ckpt:
        model.load_checkpoint(args.ckpt)
    model.eval()
    model.requires_grad_(False)

    dataset_kwargs = {"pretrained_norm_stats": resolved_stats}
    dataset = instantiate(cfg.data.train, **dataset_kwargs)
    return model, dataset, resolved_stats


def _select_indices(total: int, *, stride: int, num_shards: int, shard_index: int, max_samples: int | None):
    indices = list(range(0, total, max(int(stride), 1)))
    indices = indices[int(shard_index)::max(int(num_shards), 1)]
    if max_samples is not None:
        indices = indices[: int(max_samples)]
    return indices


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="libero_dual_regime_fused_2cam224_1e-4")
    ap.add_argument("--backbone-kind", choices=["idm"], default="idm")
    ap.add_argument("--ckpt", required=True, help="dual-regime UNCOND+IDM checkpoint")
    ap.add_argument("--allow-legacy-checkpoint", action="store_true",
                    help="allow a pre-provenance checkpoint after manual verification")
    ap.add_argument("--dataset-stats", default=None,
                    help="dataset_stats.json matching the checkpoint's training run "
                         "(defaults to the task config's pretrained_norm_stats)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="must match the online gate WAM dtype",
    )
    ap.add_argument("--inference-steps", type=int, default=20)
    ap.add_argument("--sigma-shift", type=float, default=None)
    ap.add_argument("--num-video-frames", type=int, default=None,
                    help="defaults to the dataset's derived sampled-video length")
    ap.add_argument("--cost-table", default=None, help="cost YAML from profile_wam_modes.py")
    # sampling
    ap.add_argument("--stride", type=int, default=20, help="take every Nth dataset index")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=None, help="cap per shard (smoke runs)")
    ap.add_argument("--num-seeds", type=int, default=1,
                    help="seeds per (sample, mode); >1 averages inference stochasticity")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--skip-padded", action="store_true", default=False,
                    help="drop samples whose exec window has any padded action step (default keeps valid prefixes)")
    ap.add_argument("--no-skip-padded", dest="skip_padded", action="store_false")
    # labeling knobs (stored curves allow offline relabeling later)
    ap.add_argument("--metric", choices=["l1", "l2"], default="l1")
    ap.add_argument("--exec-horizon", type=int, default=None,
                    help="score only the first N action steps (set = eval replan_steps)")
    ap.add_argument("--tol-rel", type=float, default=0.1)
    ap.add_argument("--tol-abs", type=float, default=0.02)
    # output
    ap.add_argument("--out", default="data/gate_oracle/labels",
                    help="output dir; writes <out>/shard_<i>_of_<n>.pt")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")

    import torch

    from fastwam.adaptive_gate import (
        WAMModeAdapter,
        all_mode_errors_finite,
        chunk_errors_from_steps,
        compose_group_id,
        compute_mode_step_errors,
        label_distribution,
        quality_metadata,
        select_cheapest_near_best,
        write_label_shard,
        WORLD_FEAT_LAYOUT,
        TEXT_FEAT_LAYOUT,
    )

    model, dataset, resolved_stats = _build_model_and_dataset(args)
    derived_video_frames = (
        (int(dataset.num_frames) - 1) // int(dataset.action_video_freq_ratio) + 1
    )
    if args.num_video_frames is None:
        args.num_video_frames = derived_video_frames
    elif int(args.num_video_frames) != derived_video_frames:
        raise ValueError(
            f"--num-video-frames={args.num_video_frames} does not match dataset-derived "
            f"value {derived_video_frames}."
        )
    indices = _select_indices(
        len(dataset),
        stride=args.stride,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_samples=args.max_samples,
    )
    print(f"dataset={len(dataset)} samples -> shard {args.shard_index}/{args.num_shards}: {len(indices)} states")

    # action_horizon is fixed by the data (num_frames - 1); read it per sample.
    dataset_stats_fingerprint = _sha256_file(resolved_stats)
    adapter = WAMModeAdapter(
        model,
        backbone_kind=args.backbone_kind,
        task=args.task,
        num_video_frames=args.num_video_frames,
        generation_horizon=int(dataset.num_frames) - 1,
        inference_steps=args.inference_steps,
        sigma_shift=args.sigma_shift,
        cost_table_path=args.cost_table,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
        context_len=int(dataset.context_len),
        dataset_stats_fingerprint=dataset_stats_fingerprint,
    )

    records: dict[str, list[torch.Tensor]] = {k: [] for k in (
        "world_feat", "text_feat", "proprio", "label", "chunk_err", "step_l1",
        "step_l2", "valid_steps", "sample_idx", "group_id", "best_err", "idm_err",
        "idm_regret",
    )}
    kept = dropped = nonfinite_dropped = 0
    for n, idx in enumerate(indices):
        try:
            sample = dataset.get_strict(idx)
        except Exception as exc:
            raise RuntimeError(f"strict oracle read failed for dataset index {idx}") from exc
        gt_action = sample["action"].float()                 # [T, A] normalized
        if int(gt_action.shape[0]) != adapter.generation_horizon:
            raise ValueError(
                f"dataset index {idx} action horizon {gt_action.shape[0]} does not match "
                f"configured generation horizon {adapter.generation_horizon}."
            )
        valid = ~sample["action_is_pad"].bool()              # [T]
        exec_window = valid if args.exec_horizon is None else valid[: args.exec_horizon]
        if args.skip_padded and not bool(exec_window.all()):
            dropped += 1
            continue
        input_image = sample["video"][:, 0].unsqueeze(0)     # [1, 3, H, W], [-1, 1]
        proprio = sample["proprio"][0].unsqueeze(0).float()  # [1, P] normalized
        context = sample["context"].unsqueeze(0)             # [1, L, D]
        context_mask = sample["context_mask"].unsqueeze(0)   # [1, L]

        seeds = [args.seed_base + idx * args.num_seeds + k for k in range(args.num_seeds)]
        errs = compute_mode_step_errors(
            adapter,
            input_image=input_image,
            gt_action=gt_action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            seeds=seeds,
        )
        if not all_mode_errors_finite(errs["step_l1"], errs["step_l2"]):
            dropped += 1
            nonfinite_dropped += 1
            continue
        step_err = errs["step_l1"] if args.metric == "l1" else errs["step_l2"]
        chunk_err, has_valid = chunk_errors_from_steps(
            step_err, valid, exec_horizon=args.exec_horizon
        )
        if not all_mode_errors_finite(chunk_err):
            dropped += 1
            nonfinite_dropped += 1
            continue
        if not bool(has_valid):
            dropped += 1
            continue
        label = select_cheapest_near_best(
            chunk_err, costs=errs["costs"], tol_abs=args.tol_abs, tol_rel=args.tol_rel
        )
        quality = quality_metadata(chunk_err)

        records["world_feat"].append(errs["world_feat"])
        records["text_feat"].append(errs["text_feat"])
        records["proprio"].append(proprio.squeeze(0).float().cpu())
        records["label"].append(label.reshape(()).long())
        records["chunk_err"].append(chunk_err.float())
        records["step_l1"].append(errs["step_l1"])
        records["step_l2"].append(errs["step_l2"])
        records["valid_steps"].append(valid)
        records["sample_idx"].append(torch.tensor(int(idx), dtype=torch.long))
        group_value = compose_group_id(
            sample.get("dataset_index", -1), sample.get("episode_index", -1)
        )
        records["group_id"].append(torch.tensor(group_value, dtype=torch.long))
        records["best_err"].append(quality["best_err"].reshape(()))
        records["idm_err"].append(quality["idm_err"].reshape(()))
        records["idm_regret"].append(quality["idm_regret"].reshape(()))
        kept += 1
        if args.log_every and kept % args.log_every == 0:
            labels_so_far = torch.stack(records["label"])
            print(f"[{n + 1}/{len(indices)}] kept={kept} dropped={dropped} "
                  f"dist={label_distribution(labels_so_far)}")

    if kept == 0:
        raise SystemExit("no usable samples (everything padded/dropped) — nothing to write.")

    data = {k: torch.stack(v) for k, v in records.items()}
    meta = {
        "task": args.task,
        "backbone_kind": args.backbone_kind,
        "ckpt": os.path.abspath(args.ckpt),
        "dataset_stats": resolved_stats,
        "inference_steps": args.inference_steps,
        "solver_contract": adapter.solver_contract,
        "solver_fingerprint": adapter.solver_fingerprint,
        "context_len": int(dataset.context_len),
        "model_dtype": str(model.torch_dtype),
        "num_video_frames": args.num_video_frames,
        "cost_table": dict(adapter.cost_table),
        "metric": args.metric,
        "exec_horizon": args.exec_horizon,
        "tol_rel": args.tol_rel,
        "tol_abs": args.tol_abs,
        "num_seeds": args.num_seeds,
        "seed_base": args.seed_base,
        "stride": args.stride,
        "max_samples": args.max_samples,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "skip_padded": args.skip_padded,
        "world_feat_dim": int(data["world_feat"].shape[-1]),
        "world_feat_layout": WORLD_FEAT_LAYOUT,
        "text_feat_dim": int(data["text_feat"].shape[-1]),
        "text_feat_layout": TEXT_FEAT_LAYOUT,
        "proprio_dim": int(data["proprio"].shape[-1]),
        "action_horizon": int(data["valid_steps"].shape[-1]),
        "num_dropped": dropped,
        "num_nonfinite_dropped": nonfinite_dropped,
        "group_split_available": bool((data["group_id"] >= 0).all()),
        "group_id_layout": "dataset_index_u31_episode_index_u32_v1",
        "ckpt_fingerprint": model._loaded_checkpoint_fingerprint,
        "ckpt_file_sha256": _sha256_file(args.ckpt),
        "dataset_stats_fingerprint": dataset_stats_fingerprint,
    }
    out_path = os.path.join(args.out, f"shard_{args.shard_index}_of_{args.num_shards}.pt")
    write_label_shard(out_path, data=data, meta=meta)
    print(
        f"kept={kept} dropped={dropped} nonfinite_dropped={nonfinite_dropped} "
        f"label distribution: {label_distribution(data['label'])}"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
