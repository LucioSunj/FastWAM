"""Real demonstration export and full-window sampling over the pinned LeRobot reader."""

from __future__ import annotations

import json
from copy import deepcopy
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


class RealRobotVideoDataset(RobotVideoDataset):
    """Use existing image/normalizer/text processing with only complete 32-step windows."""

    def __init__(self, **kwargs) -> None:
        if not kwargs.get("strict_sample_loading", False):
            raise ValueError(
                "Real data must not replace failed samples with random frames."
            )
        super().__init__(export_loaded_stats=False, **kwargs)
        if self.num_frames != 33 or self.global_stride != 1:
            raise ValueError(
                "Real PAD datasets require 33 observations and stride one."
            )
        boundaries = self.lerobot_dataset.episode_data_index
        self.sample_indices = [
            index
            for start, end in zip(
                boundaries["from"].tolist(), boundaries["to"].tolist()
            )
            for index in range(start, max(start, end - 32))
        ]
        self.discarded_tail_windows = len(self.lerobot_dataset) - len(
            self.sample_indices
        )
        if not self.sample_indices:
            raise ValueError("No complete 32-action/33-observation windows remain.")

    @property
    def global_stride(self) -> int:
        return self.lerobot_dataset.global_sample_stride

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _get(self, idx):
        sample = super()._get(self.sample_indices[idx])
        if bool(sample["action_is_pad"].any()) or bool(sample["image_is_pad"].any()):
            raise RuntimeError(
                "A complete real-robot window unexpectedly contains padding."
            )
        return sample


def _group_splits(sessions: list[dict]) -> dict[str, str]:
    """Keep every connected session/layout group entirely within one split."""
    groups: list[list[dict]] = []
    for session in sessions:
        related = [
            g
            for g in groups
            if any(
                s["session_id"] == session["session_id"]
                or s["layout_id"] == session["layout_id"]
                for s in g
            )
        ]
        merged = [session]
        for group in related:
            merged.extend(group)
            groups.remove(group)
        groups.append(merged)
    if len(groups) < 3:
        raise ValueError(
            "Train/validation/test require at least three disjoint session/layout groups."
        )
    order = np.random.default_rng(42).permutation(len(groups))
    result = {}
    for rank, index in enumerate(order):
        split = "test" if rank == 0 else "validation" if rank == 1 else "train"
        for session in groups[index]:
            result[session["episode_id"]] = split
    return result


def _stats(values: np.ndarray) -> dict:
    tensor = torch.from_numpy(values).float()
    return {
        "global_min": tensor.amin(0).tolist(),
        "global_max": tensor.amax(0).tolist(),
        "global_mean": tensor.mean(0).tolist(),
        "global_std": tensor.std(0, unbiased=False).tolist(),
        "global_q01": torch.quantile(tensor, 0.01, dim=0).tolist(),
        "global_q99": torch.quantile(tensor, 0.99, dim=0).tolist(),
    }


def real_dataset_config(
    directory: Path,
    *,
    cameras: list[dict],
    stats_path: Path,
    text_cache: Path,
    current_only: bool = False,
) -> dict:
    """Create a native FastWAM dataset config without LIBERO image transforms."""
    sizes = [c["resize"] for c in cameras]
    if len({tuple(s) for s in sizes}) != 1:
        raise ValueError(
            "The first real profile uses equal-size horizontally ordered camera views."
        )
    height, width = sizes[0]
    shape_meta = {
        "images": [
            {
                "key": c["name"],
                "raw_shape": [3, height, width],
                "shape": [3, height, width],
            }
            for c in cameras
        ],
        "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
        "state": [{"key": "default", "raw_shape": 8, "shape": 8}],
    }
    return {
        "_target_": "fastwam.real_robot_data.RealRobotVideoDataset",
        "dataset_dirs": [str(directory)],
        "shape_meta": shape_meta,
        "num_frames": 33,
        "global_sample_stride": 1,
        "action_video_freq_ratio": 4,
        "video_size": [height, width * len(cameras)],
        "camera_key": None,
        "val_set_proportion": 0.0,
        "is_training_set": directory.name == "train",
        "skip_padding_as_possible": False,
        "concat_multi_camera": "horizontal",
        "strict_sample_loading": True,
        "current_frame_only": current_only,
        "pretrained_norm_stats": str(stats_path),
        "text_embedding_cache_dir": str(text_cache),
        "context_len": 128,
        "processor": {
            "_target_": "fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor",
            "shape_meta": deepcopy(shape_meta),
            "num_obs_steps": 33,
            "num_image_obs_steps": 1 if current_only else 33,
            "num_output_cameras": len(cameras),
            "action_output_dim": 7,
            "proprio_output_dim": 8,
            "delta_action_dim_mask": {"default": [True] * 6 + [False]},
            "action_state_transforms": None,
            "use_stepwise_action_norm": False,
            "norm_default_mode": "min/max",
            "norm_exception_mode": None,
            "action_state_merger": {
                "_target_": "fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign"
            },
            "train_transforms": [
                {"_target_": "fastwam.datasets.lerobot.transforms.image.ToTensor"}
            ],
            "val_transforms": [
                {"_target_": "fastwam.datasets.lerobot.transforms.image.ToTensor"}
            ],
        },
    }


def write_real_robot_dataset(sessions: list[dict], output: Path) -> dict:
    """Export lossless image parquet via LeRobot v2.1, never upgrading its reader."""
    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    splits = _group_splits(sessions)
    observation = sessions[0]["observation"]
    period = sessions[0]["sample_period"]
    if any(
        s["observation"] != observation or s["sample_period"] != period
        for s in sessions
    ):
        raise ValueError(
            "All sessions in one dataset must share calibrated I/O and sampling period."
        )
    features = {
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper_open_fraction"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "tcp_x",
                "tcp_y",
                "tcp_z",
                "qx",
                "qy",
                "qz",
                "qw",
                "gripper_open_fraction",
            ],
        },
        "action_valid": {"dtype": "bool", "shape": (1,), "names": None},
    }
    for camera in observation["cameras"]:
        features[f"observation.images.{camera['name']}"] = {
            "dtype": "image",
            "shape": (*camera["resize"], 3),
            "names": ["height", "width", "channels"],
        }
    datasets = {
        split: LeRobotDataset.create(
            repo_id=f"local/pad_{split}",
            root=output / split,
            fps=1.0 / period,
            features=features,
            robot_type="franka",
            use_videos=False,
            is_compute_episode_stats_image=False,
        )
        for split in ("train", "validation", "test")
    }
    report = {
        "schema": "pad-real-dataset-v1",
        "status": "CONVERTED",
        "output": str(output),
        "source": sorted({s["source"] for s in sessions}),
        "splits": splits,
        "segments": [],
        "dropped_gap_actions": 0,
        "discarded_tail_windows": 0,
        "full_windows": 0,
        "action_offsets": list(range(32)),
        "video_offsets": list(range(0, 33, 4)),
        "sample_period": period,
        "observation": observation,
        "action_spec": sessions[0]["action_spec"],
    }
    train_actions, train_states = [], []
    for session in sessions:
        timestamps = session["timestamps"]
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError(
                "Recorded timestamps must be finite and strictly increasing."
            )
        gap_indices = np.flatnonzero(abs(np.diff(timestamps) - period) > period * 0.1)
        report["dropped_gap_actions"] += len(gap_indices)
        boundaries = [0, *(gap_indices + 1).tolist(), len(timestamps)]
        dataset = datasets[splits[session["episode_id"]]]
        for start, end in pairwise(boundaries):
            if end - start < 33:
                report["discarded_tail_windows"] += end - start
                continue
            episode_index = dataset.meta.total_episodes
            for index in range(start, end):
                valid = index + 1 < end
                frame = {
                    "action": session["actions"][index].astype(np.float32)
                    if valid
                    else np.zeros(7, dtype=np.float32),
                    "observation.state": session["states"][index].astype(np.float32),
                    "action_valid": np.array([valid], dtype=bool),
                }
                for camera in observation["cameras"]:
                    key = f"observation.images.{camera['name']}"
                    frame[key] = session["images"][camera["name"]][index]
                dataset.add_frame(
                    frame,
                    task=[
                        session["instruction"],
                        session["instruction"],
                        "recorded",
                        "recorded",
                    ],
                    timestamp=(index - start) * period,
                )
                # The vendored recorder deliberately omits image writing in add_frame.
                # Write its declared files explicitly before its native parquet embed.
                for camera in observation["cameras"]:
                    key = f"observation.images.{camera['name']}"
                    image_path = Path(dataset.episode_buffer[key][-1])
                    Image.fromarray(frame[key]).save(image_path, format="PNG")
            dataset.save_episode(raw_file_name=session["source_path"])
            windows = end - start - 32
            report["segments"].append(
                {
                    "episode_id": session["episode_id"],
                    "split": splits[session["episode_id"]],
                    "segment_index": episode_index,
                    "start": start,
                    "end": end,
                    "full_windows": windows,
                    "elapsed_timestamps": timestamps[start:end].tolist(),
                }
            )
            report["full_windows"] += windows
            report["discarded_tail_windows"] += 32
            if splits[session["episode_id"]] == "train":
                train_actions.append(session["actions"][start : end - 1])
                train_states.append(session["states"][start:end])
    if not train_actions or any(
        ds.meta.total_episodes == 0 for ds in datasets.values()
    ):
        raise ValueError(
            "Every split must contain a complete, temporally aligned 32-step window."
        )
    stats = {
        "action": {"default": _stats(np.concatenate(train_actions))},
        "state": {"default": _stats(np.concatenate(train_states))},
    }
    (output / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    (output / "manifest.json").write_text(json.dumps(report, indent=2))
    for current_only, name in ((False, "data_idm.yaml"), (True, "data_uncond_bc.yaml")):
        data_cfg = {
            ("val" if split == "validation" else split): real_dataset_config(
                output / split,
                cameras=observation["cameras"],
                stats_path=output / "dataset_stats.json",
                text_cache=output / "text_cache",
                current_only=current_only,
            )
            for split in datasets
        }
        (output / name).write_text(yaml.safe_dump(data_cfg, sort_keys=False))
    from fastwam.real_robot_adaptation import write_adaptation_configs

    write_adaptation_configs(output)
    return report
