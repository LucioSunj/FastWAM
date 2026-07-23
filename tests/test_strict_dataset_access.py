"""Oracle sampling must never silently substitute a different dataset row."""
from __future__ import annotations

import pytest


def test_base_lerobot_get_strict_accesses_exactly_one_index():
    module = pytest.importorskip("fastwam.datasets.lerobot.base_lerobot_dataset")
    method = module.BaseLerobotDataset.get_strict

    class Store:
        def __init__(self):
            self.calls = []

        def __getitem__(self, index):
            self.calls.append(index)
            if index == 3:
                raise OSError("broken record")
            return {"raw": index}

    class Fake:
        def __init__(self):
            self.multi_dataset = Store()

        def __len__(self):
            return 10

        def _split_lerobot_sample(self, sample):
            return sample

        def _process_lerobot_sample(self, index, sample):
            return {"idx": index, **sample}

    fake = Fake()
    assert method(fake, 2) == {"idx": 2, "raw": 2}
    with pytest.raises(OSError, match="broken"):
        method(fake, 3)
    assert fake.multi_dataset.calls == [2, 3]


def test_robot_video_get_strict_sets_strict_flag():
    module = pytest.importorskip("fastwam.datasets.lerobot.robot_video_dataset")
    method = module.RobotVideoDataset.get_strict

    class Fake:
        def __len__(self):
            return 5

        def _get(self, index, *, strict=False):
            return index, strict

    assert method(Fake(), 4) == (4, True)
    with pytest.raises(IndexError):
        method(Fake(), 5)


def test_pretrained_dataset_stats_are_persisted_byte_exactly(tmp_path):
    module = pytest.importorskip("fastwam.datasets.lerobot.robot_video_dataset")
    source = tmp_path / "source.json"
    target = tmp_path / "output" / "dataset_stats.json"
    target.parent.mkdir()
    payload = b'{\n  "value": 1\n}\n'
    source.write_bytes(payload)

    module._copy_pretrained_dataset_stats(source, target)

    assert target.read_bytes() == payload
    module._copy_pretrained_dataset_stats(target, target)
    assert target.read_bytes() == payload
