"""Generate self-supervised ORACLE mode labels for the adaptive-prediction gate (M3).

For each sampled state of the raw VLA training set, run the FROZEN dual-regime
WAM once per mode {SKIP, LATENT, FULL} (paired seeds), measure each mode's action
error against the dataset's ground-truth chunk, and label the state with the
CHEAPEST SUFFICIENT mode (see `fastwam.adaptive_gate.oracle`). No annotation:
the targets come from data the VLA dataset already contains.

The output shards feed the gate's BC warm-start (SFT) in RLinf:
    RLinf/examples/embodiment/train_gate_bc.py --labels 'labels/*.pt' ...

HEAVY: each sample costs ~num_seeds * (cost(SKIP)+cost(LATENT)+cost(FULL))
~= 1.5x num_seeds FULL inferences. Subsample with --stride/--max-samples and
shard across GPUs with --num-shards/--shard-index (disjoint strided slices).

Example (LIBERO, joint backbone, 4 GPUs -> 4 shards):
  cd FastWAM
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/generate_gate_oracle_labels.py \
      --task libero_metric_adaptive_joint_2cam224_1e-4 --backbone-kind joint \
      --ckpt /path/to/dual_regime_joint.pt \
      --dataset-stats /path/to/dataset_stats.json \
      --stride 20 --num-shards 4 --shard-index $i \
      --exec-horizon 10 --out data/gate_oracle/libero_joint &
  done; wait

Tolerances default to tol-rel=0.1, tol-abs=0.02 (normalized action units), but the
stored per-step error curves allow OFFLINE relabeling with different knobs via
`fastwam.adaptive_gate.relabel_from_steps` — no WAM re-run.
"""
from __future__ import annotations

import argparse
import os


def _build_model_and_dataset(args):
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    with initialize_config_dir(version_base="1.3", config_dir=configs_dir):
        cfg = compose(config_name="train", overrides=[f"task={args.task}"])

    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=args.device)
    if args.ckpt:
        model.load_checkpoint(args.ckpt)
    model.eval()
    model.requires_grad_(False)

    dataset_kwargs = {}
    if args.dataset_stats:
        dataset_kwargs["pretrained_norm_stats"] = args.dataset_stats
    dataset = instantiate(cfg.data.train, **dataset_kwargs)
    return model, dataset


def _select_indices(total: int, *, stride: int, num_shards: int, shard_index: int, max_samples: int | None):
    indices = list(range(0, total, max(int(stride), 1)))
    indices = indices[int(shard_index)::max(int(num_shards), 1)]
    if max_samples is not None:
        indices = indices[: int(max_samples)]
    return indices


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="libero_metric_adaptive_joint_2cam224_1e-4")
    ap.add_argument("--backbone-kind", choices=["joint", "idm"], default="joint")
    ap.add_argument("--ckpt", required=True, help="dual-regime checkpoint (LATENT/FULL need it)")
    ap.add_argument("--dataset-stats", default=None,
                    help="dataset_stats.json matching the checkpoint's training run "
                         "(defaults to the task config's pretrained_norm_stats)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k-lo", type=int, default=4)
    ap.add_argument("--k-hi", type=int, default=20)
    ap.add_argument("--num-video-frames", type=int, default=9)
    ap.add_argument("--cost-table", default=None, help="cost YAML from profile_wam_modes.py")
    # sampling
    ap.add_argument("--stride", type=int, default=20, help="take every Nth dataset index")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=None, help="cap per shard (smoke runs)")
    ap.add_argument("--num-seeds", type=int, default=1,
                    help="seeds per (sample, mode); >1 averages inference stochasticity")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--skip-padded", action="store_true", default=True,
                    help="drop samples whose exec window has any padded action step")
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
        chunk_errors_from_steps,
        compute_mode_step_errors,
        label_distribution,
        select_cheapest_sufficient,
        write_label_shard,
    )

    model, dataset = _build_model_and_dataset(args)
    indices = _select_indices(
        len(dataset),
        stride=args.stride,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_samples=args.max_samples,
    )
    print(f"dataset={len(dataset)} samples -> shard {args.shard_index}/{args.num_shards}: {len(indices)} states")

    # action_horizon is fixed by the data (num_frames - 1); read it per sample.
    adapter = WAMModeAdapter(
        model,
        backbone_kind=args.backbone_kind,
        num_video_frames=args.num_video_frames,
        action_horizon=1,  # overridden per call with the GT chunk length
        k_lo=args.k_lo,
        k_hi=args.k_hi,
        cost_table_path=args.cost_table,
    )

    records: dict[str, list[torch.Tensor]] = {k: [] for k in (
        "world_feat", "proprio", "label", "chunk_err", "step_l1", "step_l2", "valid_steps", "sample_idx",
    )}
    kept = dropped = 0
    for n, idx in enumerate(indices):
        sample = dataset[idx]
        gt_action = sample["action"].float()                 # [T, A] normalized
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
        step_err = errs["step_l1"] if args.metric == "l1" else errs["step_l2"]
        chunk_err, has_valid = chunk_errors_from_steps(
            step_err, valid, exec_horizon=args.exec_horizon
        )
        if not bool(has_valid):
            dropped += 1
            continue
        label = select_cheapest_sufficient(chunk_err, tol_abs=args.tol_abs, tol_rel=args.tol_rel)

        records["world_feat"].append(errs["world_feat"])
        records["proprio"].append(proprio.squeeze(0).float().cpu())
        records["label"].append(label.reshape(()).long())
        records["chunk_err"].append(chunk_err.float())
        records["step_l1"].append(errs["step_l1"])
        records["step_l2"].append(errs["step_l2"])
        records["valid_steps"].append(valid)
        records["sample_idx"].append(torch.tensor(int(idx), dtype=torch.long))
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
        "dataset_stats": args.dataset_stats,
        "k_lo": args.k_lo,
        "k_hi": args.k_hi,
        "num_video_frames": args.num_video_frames,
        "cost_table": dict(adapter.cost_table),
        "metric": args.metric,
        "exec_horizon": args.exec_horizon,
        "tol_rel": args.tol_rel,
        "tol_abs": args.tol_abs,
        "num_seeds": args.num_seeds,
        "seed_base": args.seed_base,
        "stride": args.stride,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "skip_padded": args.skip_padded,
        "world_feat_dim": int(data["world_feat"].shape[-1]),
        "proprio_dim": int(data["proprio"].shape[-1]),
        "action_horizon": int(data["valid_steps"].shape[-1]),
        "num_dropped": dropped,
    }
    out_path = os.path.join(args.out, f"shard_{args.shard_index}_of_{args.num_shards}.pt")
    write_label_shard(out_path, data=data, meta=meta)
    print(f"kept={kept} dropped={dropped} label distribution: {label_distribution(data['label'])}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
