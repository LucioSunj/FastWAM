#!/usr/bin/env python3
"""Resolve the exact episode-excluded 10-epoch E1-P2 step contract."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.adaptive_gate.sdr_contracts import artifact_record, atomic_json
from fastwam.adaptive_gate.sdr_validation import (
    collect_episode_records,
    episodes_for_manifest_split,
    read_manifest,
    validate_validation_manifest,
)
from fastwam.utils.misc import register_work_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_config).expanduser().resolve()
    output_path = Path(args.output_config).expanduser().resolve()
    manifest_path = Path(args.validation_manifest).expanduser().resolve()
    stats_path = Path(args.dataset_stats).expanduser().resolve()
    register_work_dir(output_path.parent)
    cfg = OmegaConf.load(input_path)
    if int(cfg.num_epochs) != 10:
        raise ValueError("Formal step planning requires num_epochs=10.")
    if int(cfg.batch_size) != 1 or int(cfg.gradient_accumulation_steps) != 64:
        raise ValueError("Formal step planning requires microbatch=1 and grad_accum=64.")
    if int(args.world_size) != 1:
        raise ValueError("This formal contract is preregistered for world_size=1.")
    if str(cfg.data.train.episode_split_manifest) != str(manifest_path):
        raise ValueError("Formal config does not name the locked validation manifest.")
    if str(cfg.data.train.manifest_split) != "train":
        raise ValueError("Formal dataset must use the manifest train split.")

    manifest = read_manifest(manifest_path)
    dataset_dirs = [
        str(Path(path).expanduser().resolve())
        for path in cfg.data.train.dataset_dirs
    ]
    validate_validation_manifest(
        manifest,
        dataset_dirs=dataset_dirs,
        dataset_stats=stats_path,
    )
    dataset = instantiate(cfg.data.train)
    expected_train_episodes = episodes_for_manifest_split(
        dataset_dirs=dataset_dirs,
        manifest=manifest,
        split="train",
    )
    subdatasets = dataset.lerobot_dataset.multi_dataset._datasets
    if len(subdatasets) != len(dataset_dirs):
        raise ValueError("Instantiated LeRobot dataset count changed.")
    for raw_dir, subdataset in zip(dataset_dirs, subdatasets):
        observed = list(subdataset.episodes or ())
        expected = expected_train_episodes[raw_dir]
        if observed != expected:
            raise ValueError(
                "Instantiated train episodes do not match manifest exclusion: "
                f"dataset={raw_dir}."
            )
    train_episode_keys = {
        (dataset_index, episode_index)
        for dataset_index, raw_dir in enumerate(dataset_dirs)
        for episode_index in expected_train_episodes[raw_dir]
    }
    expected_train_frames = sum(
        int(record["episode_length"])
        for record in collect_episode_records(dataset_dirs)
        if (
            int(record["dataset_index"]),
            int(record["episode_index"]),
        )
        in train_episode_keys
    )
    train_samples = int(len(dataset))
    if train_samples != expected_train_frames:
        raise ValueError(
            "Instantiated train frame count does not match selected episode "
            f"metadata: observed={train_samples}, expected={expected_train_frames}."
        )
    microbatch_global = int(cfg.batch_size) * int(args.world_size)
    micro_steps_per_epoch = max(
        math.ceil(train_samples / microbatch_global),
        1,
    )
    optimizer_steps_per_epoch = max(
        math.ceil(micro_steps_per_epoch / int(cfg.gradient_accumulation_steps)),
        1,
    )
    total_optimizer_steps = optimizer_steps_per_epoch * int(cfg.num_epochs)

    resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved["max_steps"] = total_optimizer_steps
    resolved["run_until_step"] = None
    resolved["run_until_step_fraction"] = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False),
        encoding="utf-8",
    )
    result = {
        "schema": "fastwam-sdr-formal-step-contract-v1",
        "status": "PASS",
        "plus_full_used": False,
        "input_config": artifact_record(input_path),
        "resolved_config": artifact_record(output_path),
        "validation_manifest": artifact_record(manifest_path),
        "dataset_stats": artifact_record(stats_path),
        "train_sequence_samples": train_samples,
        "train_episode_count": len(train_episode_keys),
        "held_out_episode_count": len(manifest["validation_episodes"]),
        "train_validation_episode_overlap": 0,
        "world_size": 1,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 64,
        "global_batch_size": 64,
        "micro_steps_per_epoch": micro_steps_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "num_epochs": 10,
        "total_optimizer_steps": total_optimizer_steps,
    }
    atomic_json(args.out, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
