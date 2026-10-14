"""CPU storage and BC input tests with real tiny parquet and video files."""

import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from fastwam.datasets.lerobot.easywam_dataset import (
    EasyWAMBaseLerobotDataset,
    EasyWAMDatasetMetadata,
    EasyWAMLeRobotDataset,
    EasyWAMMultiLeRobotDataset,
    EasyWAMRobotVideoDataset,
)
from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.datasets.lerobot.transforms.image import ToTensor
from omegaconf import OmegaConf

CAMERAS = ("observation.images.image", "observation.images.wrist_image")


def _parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _video(path, frame_ids, *, camera):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in frame_ids:
            pixels = np.full((16, 16, 3), 20 + index * 12 + camera * 20, np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _corpus(root: Path, *, version="v3.0", task_column=False):
    (root / "meta").mkdir(parents=True)
    features = {
        key: {
            "dtype": "video",
            "shape": [16, 16, 3],
            "names": ["height", "width", "channel"],
        }
        for key in CAMERAS
    }
    for key, dtype, shape in (
        ("action", "float32", [2]),
        ("observation.state", "float32", [3]),
        ("timestamp", "float32", [1]),
        ("episode_index", "int64", [1]),
        ("frame_index", "int64", [1]),
        ("task_index", "int64", [1]),
        ("index", "int64", [1]),
    ):
        features[key] = {"dtype": dtype, "shape": shape, "names": None}
    info = {
        "codebase_version": version,
        "fps": 10,
        "features": features,
        "total_episodes": 3,
        "total_frames": 12,
        "total_tasks": 1,
        "robot_type": "fixture",
        "chunks_size": 1000,
        "total_chunks": 1,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    rows = [
        {
            "action": [float(index), float(index + 1)],
            "observation.state": [float(index), float(index + 1), float(index + 2)],
            "timestamp": (index % 4) / 10,
            "episode_index": index // 4,
            "frame_index": index % 4,
            "task_index": 7,
            "index": index,
        }
        for index in range(12)
    ]
    if version == "v3.0":
        tasks = pd.DataFrame({"task_index": [7]}, index=pd.Index(["pick the block"]))
        if task_column:
            tasks = tasks.reset_index(names="task")
        tasks.to_parquet(root / "meta/tasks.parquet")
        episodes = []
        for index in range(3):
            episode = {
                "episode_index": index,
                "length": 4,
                "dataset_from_index": index * 4,
                "dataset_to_index": (index + 1) * 4,
                "data/chunk_index": 0,
                "data/file_index": index // 2,
                "tasks": ["pick the block"],
            }
            for camera in CAMERAS:
                episode.update(
                    {
                        f"videos/{camera}/chunk_index": 0,
                        f"videos/{camera}/file_index": 0,
                        f"videos/{camera}/from_timestamp": index * 0.4,
                        f"videos/{camera}/to_timestamp": (index + 1) * 0.4,
                    }
                )
            episodes.append(episode)
        _parquet(
            root / "meta/episodes/chunk-000/file-000.parquet",
            [episodes[2], episodes[0]],
        )
        _parquet(root / "meta/episodes/chunk-001/file-000.parquet", [episodes[1]])
        _parquet(root / "data/chunk-000/file-000.parquet", rows[:8])
        _parquet(root / "data/chunk-000/file-001.parquet", rows[8:])
        for camera, key in enumerate(CAMERAS):
            _video(
                root / f"videos/{key}/chunk-000/file-000.mp4", range(12), camera=camera
            )
    else:
        info["data_path"] = (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        info["video_path"] = (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        (root / "meta/tasks.jsonl").write_text(
            json.dumps({"task_index": 7, "task": "pick the block"}) + "\n"
        )
        (root / "meta/episodes.jsonl").write_text(
            "".join(
                json.dumps(
                    {"episode_index": index, "length": 4, "tasks": ["pick the block"]}
                )
                + "\n"
                for index in range(3)
            )
        )
        (root / "meta/episodes_stats.jsonl").write_text(
            "".join(
                json.dumps({"episode_index": index, "stats": {}}) + "\n"
                for index in range(3)
            )
        )
        for index in range(3):
            _parquet(
                root / f"data/chunk-000/episode_{index:06d}.parquet",
                rows[index * 4 : (index + 1) * 4],
            )
            for camera, key in enumerate(CAMERAS):
                _video(
                    root / f"videos/chunk-000/{key}/episode_{index:06d}.mp4",
                    range(index * 4, (index + 1) * 4),
                    camera=camera,
                )
    (root / "meta/info.json").write_text(json.dumps(info))
    return root


@pytest.fixture
def v3(tmp_path):
    return _corpus(tmp_path / "v3")


def test_v3_shuffled_split_keeps_parquet_rows_video_offsets_and_padding(v3):
    deltas = {key: [0, 0.1, 0.2] for key in (*CAMERAS, "action", "observation.state")}
    dataset = EasyWAMLeRobotDataset(
        str(v3), v3, episodes=[2, 0], delta_timestamps=deltas, video_backend="pyav"
    )
    assert len(dataset) == 8
    assert dataset[0]["index"].item() == 8
    assert dataset[4]["index"].item() == 0
    end = dataset[3]
    assert end["action"][:, 0].tolist() == [11, 11, 11]
    assert end["action_is_pad"].tolist() == [False, True, True]
    assert end["task"] == "pick the block"
    for camera, key in enumerate(CAMERAS):
        expected = (20 + 11 * 12 + camera * 20) / 255
        torch.testing.assert_close(
            end[key].mean(), torch.tensor(expected), atol=0.02, rtol=0
        )
    assert dataset.get_episode_data(0)["index"].tolist() == [8, 9, 10, 11]
    assert dataset.get_episode_data(1)["index"].tolist() == [0, 1, 2, 3]


def test_mixed_v21_v3_preserves_legacy_samples_and_episode_data(v3, tmp_path):
    v21 = _corpus(tmp_path / "v21", version="v2.1")
    deltas = {"action": [0, 0.1]}
    legacy = LeRobotDataset(
        str(v21), v21, episodes=[1], delta_timestamps=deltas, video_backend="pyav"
    )
    mixed = EasyWAMMultiLeRobotDataset(
        [str(v21), str(v3)],
        episodes={str(v21): [1], str(v3): [2, 0]},
        delta_timestamps=deltas,
        video_backend="pyav",
    )
    assert mixed.num_episodes == 3
    assert mixed.num_frames == 12
    for key in ("action", "observation.state", *CAMERAS):
        torch.testing.assert_close(mixed[1][key], legacy[1][key])
    assert mixed[4]["index"].item() == 8
    assert mixed[4]["dataset_index"].item() == 1
    assert mixed.get_episode_data(1)["index"].tolist() == [8, 9, 10, 11]


def _shape_meta():
    return {
        "images": [
            {
                "key": key.rsplit(".", 1)[-1],
                "raw_shape": [3, 16, 16],
                "shape": [3, 16, 16],
            }
            for key in CAMERAS
        ],
        "action": [{"key": "default", "raw_shape": 2, "shape": 2}],
        "state": [{"key": "default", "raw_shape": 3, "shape": 3}],
    }


def test_bc_dataset_reuses_processor_camera_prompt_and_masked_cache(
    v3, tmp_path, monkeypatch
):
    from fastwam.datasets.lerobot import robot_video_dataset

    monkeypatch.setattr(robot_video_dataset.misc, "get_work_dir", lambda: str(tmp_path))
    stats = {
        key: {"default": {"global_min": [0.0] * width, "global_max": [20.0] * width}}
        for key, width in (("action", 2), ("state", 3))
    }
    stats_path = tmp_path / "parent_stats.json"
    stats_path.write_text(json.dumps(stats))
    prompt = DEFAULT_PROMPT.format(task="pick the block")
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    cache = tmp_path / "text"
    cache.mkdir()
    context = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    mask = torch.tensor([True, True, False])
    torch.save(
        {"context": context, "mask": mask}, cache / f"{digest}.text_len3.wan22ti2v5b.pt"
    )
    shape = _shape_meta()
    processor = FastWAMProcessor(
        shape_meta=OmegaConf.create(shape),
        num_obs_steps=5,
        num_image_obs_steps=1,
        num_output_cameras=2,
        action_output_dim=2,
        proprio_output_dim=3,
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="min/max",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=[ToTensor()],
        val_transforms=[ToTensor()],
    )
    dataset = EasyWAMRobotVideoDataset(
        dataset_dirs=[str(v3)],
        shape_meta=OmegaConf.create(shape),
        num_frames=5,
        video_size=[16, 32],
        processor=processor,
        text_embedding_cache_dir=str(cache),
        context_len=3,
        pretrained_norm_stats=str(stats_path),
        val_set_proportion=0,
        current_frame_only=True,
        strict_sample_loading=True,
        text_padding="masked",
    )
    sample = dataset[8]
    assert sample["video"].shape == (3, 1, 16, 32)
    assert sample["action"].shape == (4, 2)
    assert sample["proprio"].shape == (4, 3)
    assert sample["prompt"] == prompt
    assert sample["action_is_pad"].tolist() == [False] * 4
    torch.testing.assert_close(sample["action"][0], torch.tensor([-0.2, -0.1]))
    torch.testing.assert_close(sample["context"], context)
    assert torch.equal(sample["context_mask"], mask)
    assert sample["video"][..., 16:].mean() > sample["video"][..., :16].mean()


def test_seed42_episode_split_is_disjoint_and_complete(v3):
    kwargs = {
        "dataset_dirs": [str(v3)],
        "shape_meta": _shape_meta(),
        "obs_size": 5,
        "action_size": 4,
        "image_obs_size": 1,
        "val_set_proportion": 0.1,
    }
    train = EasyWAMBaseLerobotDataset(**kwargs, is_training_set=True)
    validation = EasyWAMBaseLerobotDataset(**kwargs, is_training_set=False)
    expected = np.arange(3)
    np.random.default_rng(42).shuffle(expected)
    assert train.multi_dataset._datasets[0].episodes == expected[:2].tolist()
    assert validation.multi_dataset._datasets[0].episodes == expected[2:].tolist()
    assert len(train) + len(validation) == 12


def test_v3_task_column_and_missing_local_metadata(tmp_path):
    path = _corpus(tmp_path / "v3", task_column=True)
    assert EasyWAMDatasetMetadata(str(path), path).tasks == {7: "pick the block"}
    with pytest.raises(FileNotFoundError):
        EasyWAMDatasetMetadata("unused/local-dataset", tmp_path / "missing")
