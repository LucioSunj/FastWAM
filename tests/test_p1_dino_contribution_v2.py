"""CPU contracts for DINO-to-action contribution v2."""

from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

from fastwam.p1_dino_contribution_v2 import (
    CausalCheckpointSelector,
    CausalSelectionThresholds,
    DependencyWarmupController,
    NegativeModeCycle,
    TaskPairedDistributedBatchSampler,
    build_frozen_causal_ledger,
    load_frozen_causal_ledger,
)
from fastwam.models.wan22.visual_contracts import NativePatchMemory
from fastwam.p1_dino_bc_full_trainer import (
    _run_acceptance_passed,
    _validate_full_config,
)
from fastwam.p1_dino_contribution_audit import _append_diagnostics
from fastwam.p1_dino_libero_inference import (
    P1DinoCompiledLiberoPolicy,
    _fixed_random_patch_memory,
    resolve_compile_cache_seed,
)


def _sampler(*, epoch: int = 0) -> TaskPairedDistributedBatchSampler:
    counts = (31, 33, 32, 32)
    task_keys = []
    episodes = []
    for task, count in enumerate(counts):
        task_keys.extend((0, task) for _ in range(count))
        episodes.extend(index % 5 for index in range(count))
    sampler = TaskPairedDistributedBatchSampler(
        task_keys=task_keys,
        episode_indices=episodes,
        batch_size=32,
        rank=0,
        world_size=1,
        seed=42,
    )
    sampler.set_epoch(epoch)
    return sampler


class _SyntheticLedgerDataset:
    def __init__(self, *, split: str) -> None:
        sources = []
        self.split = split
        self.lengths = []
        for _suite in range(4):
            episodes = []
            frames = []
            tasks = []
            states = []
            for task in range(10):
                episode = task if split == "validation" else 100 + task
                for frame in range(16):
                    episodes.append(episode)
                    frames.append(frame)
                    tasks.append(task)
                    states.append([frame / 16.0])
            source = SimpleNamespace(
                hf_dataset={
                    "episode_index": episodes,
                    "frame_index": frames,
                    "task_index": tasks,
                    "observation.state": states,
                },
                num_frames=len(episodes),
            )
            sources.append(source)
            self.lengths.append(len(episodes))
        normalizer = SimpleNamespace(
            normalizers={
                "state": {
                    "default": SimpleNamespace(
                        scale=torch.ones(1),
                        offset=torch.zeros(1),
                    )
                }
            }
        )
        multi = SimpleNamespace(
            ds_names=[f"suite_{index}" for index in range(4)],
            _datasets=sources,
        )
        robot = SimpleNamespace(
            multi_dataset=multi,
            processor=SimpleNamespace(normalizer=normalizer),
        )
        self.dataset = SimpleNamespace(lerobot_dataset=robot)

    def __len__(self) -> int:
        return sum(self.lengths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pixels = torch.zeros(2, 1, 1, 3, dtype=torch.uint8)
        flat = pixels.view(-1)
        flat[0] = 1 if self.split == "validation" else 2
        flat[1] = index & 0xFF
        flat[2] = (index >> 8) & 0xFF
        return {
            "p1_camera_pixels": pixels,
            "p1_camera_valid_mask": torch.ones(2, dtype=torch.bool),
        }


def test_frozen_ledger_uses_recorded_train_fallback_for_singletons(tmp_path) -> None:
    path = tmp_path / "ledger.json"
    payload = build_frozen_causal_ledger(
        _SyntheticLedgerDataset(split="validation"),
        path,
        negative_fallback_dataset=_SyntheticLedgerDataset(split="train"),
    )
    assert len(payload["anchors"]) == 640
    assert payload["negative_fallback_pool"]["used_anchor_count"] == 640
    assert all(value["anchor_split"] == "validation" for value in payload["anchors"])
    assert all(value["negative_split"] == "train" for value in payload["anchors"])
    assert all(
        value["anchor_episode_index"] != value["negative_episode_index"]
        and value["anchor_rgb_sha256"] != value["negative_rgb_sha256"]
        for value in payload["anchors"]
    )
    restored = load_frozen_causal_ledger(path)
    assert restored["content_sha256"] == payload["content_sha256"]


def test_task_paired_sampler_covers_each_primary_without_singletons() -> None:
    sampler = _sampler()
    batches = list(sampler)
    assert len(batches) == 4
    assert sorted(index for batch in batches for index in batch) == list(range(128))
    assert len({index for batch in batches for index in batch}) == 128

    for batch in batches:
        identities = [f"train-seed42:{index}" for index in batch]
        permutation = sampler.permutation_for_sample_identities(identities).tolist()
        assert sorted(permutation) == list(range(32))
        for source, target in enumerate(permutation):
            assert source != target
            assert sampler.task_keys[batch[source]] == sampler.task_keys[batch[target]]
            assert (
                sampler.episode_indices[batch[source]]
                != sampler.episode_indices[batch[target]]
            )

    state = sampler.state_dict()
    _sampler().validate_state_dict(state)
    assert _sampler(epoch=1).state_dict()["order_sha256"] != state["order_sha256"]
    changed = copy.deepcopy(state)
    changed["order_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="resume state/order"):
        sampler.validate_state_dict(changed)


def test_dynamic_warmup_unlocks_at_256_and_ramp_has_exact_boundaries() -> None:
    controller = DependencyWarmupController()
    for step in range(1, 256):
        controller.observe(
            correct_loss=1.0,
            negative_loss=1.2,
            global_step_after_update=step,
        )
        assert controller.reader_only
    controller.observe(
        correct_loss=1.0,
        negative_loss=1.2,
        global_step_after_update=256,
    )
    assert controller.phase == "joint"
    assert controller.warmup_end_step == 256
    assert controller.next_lora_gradient_multiplier == pytest.approx(1 / 512)
    for _ in range(511):
        controller.record_joint_update()
    assert controller.next_lora_gradient_multiplier == pytest.approx(1.0)
    controller.record_joint_update()
    assert controller.lora_ramp_progress == 512
    assert controller.next_lora_gradient_multiplier == pytest.approx(1.0)

    restored = DependencyWarmupController.from_state_dict(controller.state_dict())
    assert restored.state_dict() == controller.state_dict()


def test_dynamic_warmup_fails_closed_at_1024() -> None:
    controller = DependencyWarmupController()
    for step in range(1, 1025):
        controller.observe(
            correct_loss=1.0,
            negative_loss=1.0,
            global_step_after_update=step,
        )
    assert controller.failed
    assert controller.warmup_end_step is None


def test_negative_cycle_resume_is_exact() -> None:
    cycle = NegativeModeCycle()
    observed = []
    for _ in range(7):
        observed.append(cycle.current)
        cycle.advance()
    assert observed == [
        "task_paired",
        "task_paired",
        "task_paired",
        "off",
        "task_paired",
        "task_paired",
        "task_paired",
    ]
    restored = NegativeModeCycle.from_state_dict(cycle.state_dict())
    assert restored.current == cycle.current
    assert restored.state_dict() == cycle.state_dict()


def test_causal_selector_rejects_lower_loss_without_dependency() -> None:
    selector = CausalCheckpointSelector(
        CausalSelectionThresholds(
            validation_loss_max=0.05,
            pose_mse_max=0.04,
            gripper_mse_max=0.09,
        )
    )
    validation = {"loss_action_bc": 0.04, "mse_pose": 0.03, "mse_gripper": 0.08}
    bad = selector.assess(
        step=100,
        validation=validation,
        causal={
            "task_paired_relative_gap": 0.01,
            "off_relative_gap": 0.01,
            "positive_task_fraction": 1.0,
            "residual_hidden_p95": 0.10,
            "finite": True,
        },
        frozen_contract_unchanged=True,
    )
    assert not bad["eligible"]
    assert not selector.consider(bad)
    assert selector.best is None

    good = selector.assess(
        step=200,
        validation=validation,
        causal={
            "task_paired_relative_gap": 0.11,
            "off_relative_gap": 0.12,
            "positive_task_fraction": 0.75,
            "residual_hidden_p95": 0.10,
            "finite": True,
        },
        frozen_contract_unchanged=True,
    )
    assert selector.consider(good)
    restored = CausalCheckpointSelector.from_state_dict(selector.state_dict())
    assert restored.state_dict() == selector.state_dict()


def test_v2_hydra_profile_is_independent_and_strict() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="p1_dino_bc_dino_contribution_v2")
    _validate_full_config(cfg, world_size=4)
    assert cfg.training.dino_contribution_profile == "dino_contribution_v2"
    assert cfg.training.microbatch_size == 32
    assert cfg.training.gradient_accumulation_steps == 1
    assert cfg.training.gradient_sync == "deterministic_rank_order"
    assert cfg.training.gradient_sync_chunk_mb == 16
    assert list(cfg.training.memory_dependency.negative_cycle) == [
        "task_paired",
        "task_paired",
        "task_paired",
        "off",
    ]
    assert cfg.data.num_workers == 3
    assert cfg.data.prefetch_factor == 1

    cfg.training.reader_warmup.max_steps = 1023
    with pytest.raises(ValueError, match="warm-up/ramp"):
        _validate_full_config(cfg, world_size=4)


def test_completed_pilot_does_not_require_an_eligible_checkpoint() -> None:
    common = {
        "global_step": 1942,
        "stop_after_steps": 1942,
        "nonzero_update_count": 1942,
        "last_checkpoint": {"result": "PASS"},
        "best_step": None,
        "controlled_stop": False,
        "early_stopped": False,
    }
    assert not _run_acceptance_passed(**common)
    assert _run_acceptance_passed(**common, eligibility_required=False)


def test_compile_cache_seed_resolves_current_and_legacy_workers(tmp_path) -> None:
    current = tmp_path / "current" / "torchinductor" / "identity" / "worker_3"
    current.mkdir(parents=True)
    (current / "kernel.py").write_text("compiled", encoding="utf-8")
    assert (
        resolve_compile_cache_seed(
            tmp_path / "current",
            cache_name="torchinductor",
            compile_identity="identity",
            worker_id=3,
        )
        == current
    )

    legacy = tmp_path / "legacy" / "triton" / "worker_3"
    legacy.mkdir(parents=True)
    (legacy / "kernel.json").write_text("compiled", encoding="utf-8")
    assert (
        resolve_compile_cache_seed(
            tmp_path / "legacy",
            cache_name="triton",
            compile_identity="identity",
            worker_id=3,
        )
        == legacy
    )

    with pytest.raises(FileNotFoundError, match="No triton cache"):
        resolve_compile_cache_seed(
            tmp_path / "missing",
            cache_name="triton",
            compile_identity="identity",
            worker_id=3,
        )


def test_diagnostics_include_layer_token_and_view_aggregates() -> None:
    destination: dict[str, list[torch.Tensor]] = defaultdict(list)
    _append_diagnostics(
        destination,
        [
            {
                "layer_index": 6,
                "camera_ids": ("main", "wrist"),
                "camera_valid_mask": torch.tensor([[True, True], [True, False]]),
                "gate_logits": torch.zeros(2, 1),
                "effective_gate": torch.ones(2, 1),
                "projected_residual_over_hidden": torch.ones(2, 3),
                "effective_residual_over_hidden": torch.ones(2, 3),
                "attention_entropy": torch.ones(2, 2, 3),
                "attention_top1": torch.ones(2, 2, 3),
                "attention_top5": torch.ones(2, 2, 3),
                "effective_patch_count": torch.ones(2, 2, 3),
            }
        ],
    )
    assert "layer_6/action_token_2/effective_residual_over_hidden" in destination
    assert "layer_6/view_main/action_token_1/attention_entropy" in destination
    assert len(destination["layer_6/view_wrist/action_token_0/attention_top1"][0]) == 1


@pytest.mark.parametrize(
    ("memory_mode", "inference_calls", "encoder_calls", "expected_pass"),
    [
        ("correct", 9, 10, True),
        ("correct", 9, 9, False),
        ("random_tensor", 9, 10, True),
        ("random_vit", 9, 10, True),
        ("off", 9, 0, True),
        ("off", 9, 1, False),
    ],
)
def test_libero_dino_call_count_audit_fails_closed(
    memory_mode: str,
    inference_calls: int,
    encoder_calls: int,
    expected_pass: bool,
) -> None:
    wrapper = object.__new__(P1DinoCompiledLiberoPolicy)
    torch.nn.Module.__init__(wrapper)
    wrapper.memory_mode = memory_mode
    wrapper.memory_enabled = memory_mode != "off"
    wrapper.random_patch_seed = 2026081321
    wrapper.compile_enabled = True
    wrapper._parity_report = object()
    wrapper._inference_call_count = inference_calls
    wrapper._visual_memory_encode_count = encoder_calls
    assert wrapper.dino_call_audit["contract_passed"] is expected_pass


def test_fixed_random_patch_memory_is_deterministic_and_preserves_masks() -> None:
    valid = torch.tensor([[True, False]])
    patch_valid = valid.unsqueeze(-1).expand(-1, -1, 196)
    tokens = torch.ones(1, 2, 196, 384)
    tokens[:, 1] = 0
    memory = NativePatchMemory(
        tokens=tokens,
        patch_valid_mask=patch_valid,
        camera_valid_mask=valid,
        camera_ids=("main", "wrist"),
        grid=(14, 14),
        source_revision="test",
        weights_sha256="0" * 64,
        input_contract_sha256="1" * 64,
        preprocess_sha256="2" * 64,
        output_contract_sha256="3" * 64,
        memory_contract_sha256="4" * 64,
    )
    first = _fixed_random_patch_memory(memory, seed=7)
    second = _fixed_random_patch_memory(memory, seed=7)
    assert torch.equal(first.tokens, second.tokens)
    assert torch.equal(first.patch_valid_mask, memory.patch_valid_mask)
    assert bool((first.tokens[:, 1] == 0).all())
    assert bool((first.tokens[:, 0].std(dim=-1) > 0.99).all())
