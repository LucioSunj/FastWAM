import hashlib

import pytest
import torch

from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


def test_strict_robot_video_dataset_does_not_replace_failed_sample() -> None:
    dataset = object.__new__(RobotVideoDataset)
    dataset.strict_sample_loading = True

    def fail(index):
        raise RuntimeError(f"decode failed for {index}")

    dataset._get = fail

    with pytest.raises(RuntimeError, match="decode failed for 7"):
        dataset[7]


def test_text_context_is_loaded_once_per_dataset_process(tmp_path) -> None:
    prompt = "cached task"
    context_len = 3
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache = tmp_path / f"{hashed}.t5_len{context_len}.wan22ti2v5b.pt"
    context = torch.arange(12, dtype=torch.float32).view(3, 4)
    mask = torch.tensor([True, True, False])
    torch.save({"context": context, "mask": mask}, cache)

    dataset = object.__new__(RobotVideoDataset)
    dataset.text_embedding_cache_dir = str(tmp_path)
    dataset.context_len = context_len
    dataset._text_context_cache = {}

    first = dataset._get_cached_text_context(prompt)
    cache.unlink()
    second = dataset._get_cached_text_context(prompt)

    assert first[0] is second[0]
    assert first[1] is second[1]
    expected_context = context.clone()
    expected_context[~mask] = 0.0
    assert torch.equal(first[0], expected_context)
    assert torch.equal(first[1], torch.ones_like(mask))
