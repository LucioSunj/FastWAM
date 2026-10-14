"""Local EasyWAM LeRobot v3/v2.1 inputs for the existing FastWAM BC pipeline.

Only storage lookup changes here. Episode splits, processors, normalization,
current-frame loading, camera composition and text caches belong to the existing
FastWAM dataset classes.
"""

from bisect import bisect_right
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from packaging.version import Version

from .base_lerobot_dataset import BaseLerobotDataset
from .lerobot.datasets.utils import (
    get_delta_indices,
    get_episode_data_index,
    hf_transform_to_torch,
    load_info,
    load_stats,
)
from .lerobot.datasets.video_utils import get_safe_default_codec
from .lerobot.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from .robot_video_dataset import RobotVideoDataset


class EasyWAMDatasetMetadata(LeRobotDatasetMetadata):
    """Read local v3 relational metadata or the existing v2.1 metadata."""

    def __init__(self, repo_id: str, root: str | Path) -> None:
        self.repo_id = repo_id
        self.root = Path(root)
        self.load_metadata()

    def load_metadata(self) -> None:
        self.info = load_info(self.root)
        if self._version == Version("2.1"):
            super().load_metadata()
            return
        if self._version != Version("3.0"):
            raise ValueError(
                f"EasyWAM BC supports LeRobot v2.1/v3.0, got {self._version}."
            )

        self.tasks = {}
        if self.total_tasks:
            tasks = pd.read_parquet(self.root / "meta/tasks.parquet")
            if "task" in tasks.columns:
                task_names = tasks["task"]
            elif "__index_level_0__" in tasks.columns:
                task_names = tasks["__index_level_0__"]
            elif not isinstance(tasks.index, pd.RangeIndex):
                task_names = tasks.index
            else:
                raise ValueError("LeRobot v3 task metadata has no task text.")
            self.tasks = {
                int(index): str(task)
                for index, task in zip(tasks["task_index"], task_names, strict=True)
            }
        self.task_to_task_index = {task: index for index, task in self.tasks.items()}

        files = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No LeRobot v3 episodes under {self.root}.")
        self.episodes = {}
        required = {
            "episode_index",
            "length",
            "dataset_from_index",
            "dataset_to_index",
            "data/chunk_index",
            "data/file_index",
        }
        for key in self.video_keys:
            required.update(
                f"videos/{key}/{field}"
                for field in (
                    "chunk_index",
                    "file_index",
                    "from_timestamp",
                    "to_timestamp",
                )
            )
        for path in files:
            table = pq.read_table(path)
            missing = required.difference(table.column_names)
            if missing:
                raise ValueError(
                    f"Missing LeRobot v3 episode columns: {sorted(missing)}"
                )
            for episode in table.to_pylist():
                episode_id = int(episode["episode_index"])
                if episode_id in self.episodes:
                    raise ValueError(f"Duplicate LeRobot v3 episode {episode_id}.")
                if (
                    episode["dataset_to_index"] - episode["dataset_from_index"]
                    != episode["length"]
                ):
                    raise ValueError(
                        f"Inconsistent LeRobot v3 episode {episode_id} length."
                    )
                self.episodes[episode_id] = episode
        self.episodes = dict(sorted(self.episodes.items()))
        self.episodes_stats = {}
        self.stats = load_stats(self.root) or {}

    def get_data_file_path(self, ep_index: int) -> Path:
        if self._version == Version("2.1"):
            return super().get_data_file_path(ep_index)
        episode = self.episodes[ep_index]
        return Path(
            self.data_path.format(
                chunk_index=int(episode["data/chunk_index"]),
                file_index=int(episode["data/file_index"]),
            )
        )

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        if self._version == Version("2.1"):
            return super().get_video_file_path(ep_index, vid_key)
        episode = self.episodes[ep_index]
        return Path(
            self.video_path.format(
                video_key=vid_key,
                chunk_index=int(episode[f"videos/{vid_key}/chunk_index"]),
                file_index=int(episode[f"videos/{vid_key}/file_index"]),
            )
        )


class EasyWAMLeRobotDataset(LeRobotDataset):
    """Keep logical split indices separate from v3 parquet/video offsets."""

    def __init__(
        self,
        repo_id,
        root,
        episodes=None,
        image_transforms=None,
        delta_timestamps=None,
        tolerance_s=1e-4,
        download_videos=True,
        video_backend=None,
    ):
        torch.utils.data.Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(root)
        self.meta = EasyWAMDatasetMetadata(repo_id, self.root)
        self._is_v3 = self.meta._version == Version("3.0")
        self.episodes = list(self.meta.episodes) if episodes is None else list(episodes)
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend or get_safe_default_codec()
        self.during_training = True
        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self.episodes
        )
        self._episode_ends = self.episode_data_index["to"].tolist()
        self.delta_indices = (
            get_delta_indices(delta_timestamps, self.fps)
            if delta_timestamps is not None
            else None
        )
        self.hf_dataset = self.load_hf_dataset()

    def load_hf_dataset(self):
        if not self._is_v3:
            return super().load_hf_dataset()
        files = sorted(str(path) for path in (self.root / "data").rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No LeRobot v3 data under {self.root}.")
        dataset = load_dataset("parquet", data_files=files, split="train")
        if len(dataset) != self.meta.total_frames:
            raise ValueError(
                f"LeRobot v3 frame count mismatch: {len(dataset)} != {self.meta.total_frames}."
            )
        dataset.set_transform(hf_transform_to_torch)
        return dataset

    @property
    def num_frames(self) -> int:
        return self._episode_ends[-1] if self._episode_ends else 0

    def _storage_index(self, idx: int) -> int:
        if not self._is_v3:
            return idx
        position = bisect_right(self._episode_ends, idx)
        episode = self.meta.episodes[self.episodes[position]]
        local_start = int(self.episode_data_index["from"][position])
        return int(episode["dataset_from_index"]) + idx - local_start

    def _storage_queries(self, queries):
        return {
            key: [self._storage_index(index) for index in indices]
            for key, indices in queries.items()
        }

    def _query_hf_dataset_fast(self, query_indices):
        return super()._query_hf_dataset_fast(self._storage_queries(query_indices))

    def _get_query_timestamps(self, current_ts, query_indices=None):
        return super()._get_query_timestamps(
            current_ts,
            None if query_indices is None else self._storage_queries(query_indices),
        )

    def _query_videos(self, query_timestamps, ep_idx):
        if self._is_v3:
            episode = self.meta.episodes[ep_idx]
            query_timestamps = {
                key: [
                    float(episode[f"videos/{key}/from_timestamp"]) + value
                    for value in values
                ]
                for key, values in query_timestamps.items()
            }
        return super()._query_videos(query_timestamps, ep_idx)

    def get_episode_data(self, episode_position: int) -> dict:
        """Read a selected episode without leaking adjacent rows from its shard."""
        start = int(self.episode_data_index["from"][episode_position])
        end = int(self.episode_data_index["to"][episode_position])
        selected = self.hf_dataset[
            [self._storage_index(idx) for idx in range(start, end)]
        ]
        return {
            key: torch.stack(selected[key])
            for key in self.meta.features.keys() - set(self.meta.camera_keys)
        }

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        position = bisect_right(self._episode_ends, idx)
        item = self.hf_dataset[self._storage_index(idx)]
        episode_id = int(item["episode_index"])
        if episode_id != self.episodes[position]:
            raise ValueError(f"LeRobot metadata/data episode mismatch at index {idx}.")
        queries = None
        if self.delta_indices is not None:
            queries, padding = self._get_query_indices(idx, position)
            item.update(padding)
            item.update(self._query_hf_dataset_fast(queries))
        if self.meta.video_keys and self.during_training:
            timestamps = self._get_query_timestamps(float(item["timestamp"]), queries)
            item.update(self._query_videos(timestamps, episode_id))
        if self.image_transforms is not None:
            for key in self.meta.camera_keys:
                item[key] = self.image_transforms(item[key])
        if "task_index" in item:
            item["task"] = self.meta.tasks[int(item["task_index"])]
        else:
            tasks = self.meta.episodes[episode_id].get("tasks") or []
            if len(tasks) != 1:
                raise ValueError(f"Episode {episode_id} has no unambiguous task.")
            item["task"] = str(tasks[0])
        if "subtask_annotation" in item and hasattr(self.meta, "annotations"):
            item["subtask"] = self.meta.annotations["subtask"][
                int(item["subtask_annotation"][0])
            ]
        for source, destination in (
            ("coarse_task_index", "coarse_task"),
            ("operating_hand_index", "operating_hand"),
            ("atomic_task_index", "subtask"),
        ):
            if source in item and item[source] is not None:
                item[destination] = self.meta.tasks[int(item[source])]
        return item


class EasyWAMMultiLeRobotDataset(MultiLeRobotDataset):
    """Compose v2.1 and v3 sources with the existing multi-dataset contract."""

    dataset_class = EasyWAMLeRobotDataset

    def get_episode_data(self, episode_idx: int) -> dict:
        for dataset in self._datasets:
            if episode_idx < dataset.num_episodes:
                return dataset.get_episode_data(episode_idx)
            episode_idx -= dataset.num_episodes
        raise IndexError(episode_idx)


class EasyWAMBaseLerobotDataset(BaseLerobotDataset):
    """Reuse FastWAM splitting and processing on EasyWAM storage formats."""

    metadata_class = EasyWAMDatasetMetadata
    multi_dataset_class = EasyWAMMultiLeRobotDataset


class EasyWAMRobotVideoDataset(RobotVideoDataset):
    """BC Hydra target accepting local LeRobot v3/v2.1 corpora."""

    base_dataset_class = EasyWAMBaseLerobotDataset
