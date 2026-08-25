from __future__ import annotations

from pathlib import Path

import torch

from fastwam.datasets.lerobot import robot_video_dataset as dataset_module
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


class _IdentityTransform:
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _SampleSource:
    def __init__(self, sample):
        self.sample = sample

    def __getitem__(self, index):
        del index
        return self.sample

    def __len__(self):
        return 1


def _dataset(sample, *, current_frame_image_only: bool) -> RobotVideoDataset:
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.current_frame_image_only = current_frame_image_only
    dataset.current_frame_only = False
    dataset.num_frames = 33
    dataset.lerobot_dataset = _SampleSource(sample)
    dataset.max_padding_retry = 0
    dataset.skip_padding_as_possible = False
    dataset.video_sample_indices = list(range(0, 33, 4))
    dataset.concat_multi_camera = "horizontal"
    dataset.resize_transform = _IdentityTransform()
    dataset.crop_transform = _IdentityTransform()
    dataset.normalize_transform = _IdentityTransform()
    dataset.override_instruction = None
    dataset._get_cached_text_context = lambda _instruction: (
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
        torch.ones(3, dtype=torch.bool),
    )
    return dataset


def _sample(video: torch.Tensor) -> dict:
    frames = video.shape[1]
    return {
        "pixel_values": video,
        "image_is_pad": torch.zeros(frames, dtype=torch.bool),
        "action": torch.arange(32 * 7, dtype=torch.float32).reshape(32, 7),
        "proprio": torch.arange(33 * 8, dtype=torch.float32).reshape(33, 8),
        "action_is_pad": torch.zeros(32, dtype=torch.bool),
        "proprio_is_pad": torch.zeros(33, dtype=torch.bool),
        "instruction": "pick up the block",
    }


def test_current_frame_path_preserves_policy_inputs_and_external_shape() -> None:
    full_video = torch.arange(2 * 33 * 3 * 4 * 5, dtype=torch.float32).reshape(
        2, 33, 3, 4, 5
    )
    legacy = _dataset(
        _sample(full_video.clone()),
        current_frame_image_only=False,
    )._get(0)
    optimized = _dataset(
        _sample(full_video[:, :1].clone()),
        current_frame_image_only=True,
    )._get(0)

    assert legacy["video"].shape == optimized["video"].shape == (3, 9, 4, 10)
    assert torch.equal(legacy["video"][:, :1], optimized["video"][:, :1])
    assert torch.equal(optimized["video"], optimized["video"][:, :1].repeat(1, 9, 1, 1))
    for key in (
        "action",
        "proprio",
        "context",
        "context_mask",
        "action_is_pad",
        "proprio_is_pad",
    ):
        assert torch.equal(legacy[key], optimized[key])


def test_text_context_is_loaded_once_and_keeps_wan_mask_semantics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prompt = "A video prompt"
    hashed = dataset_module.hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = tmp_path / f"{hashed}.t5_len3.wan22ti2v5b.pt"
    torch.save(
        {
            "context": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "mask": torch.tensor([True, False, True]),
        },
        cache_path,
    )
    real_load = torch.load
    calls = []

    def counted_load(*args, **kwargs):
        calls.append(args[0])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(dataset_module.torch, "load", counted_load)
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.text_embedding_cache_dir = str(tmp_path)
    dataset.context_len = 3
    dataset._text_context_cache = {}

    first_context, first_mask = dataset._get_cached_text_context(prompt)
    second_context, second_mask = dataset._get_cached_text_context(prompt)

    assert calls == [str(cache_path)]
    assert first_context.data_ptr() == second_context.data_ptr()
    assert first_mask.data_ptr() == second_mask.data_ptr()
    assert torch.equal(first_context[1], torch.zeros(4))
    assert torch.equal(first_mask, torch.ones(3, dtype=torch.bool))
