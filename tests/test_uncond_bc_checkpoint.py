import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from fastwam.adapters import (
    ActionLoRATargetGroup,
    PolicyRegime,
    RegimeLoRAConfig,
    inject_action_dit_lora,
)
from fastwam.uncond_bc_checkpoint import (
    capture_rng_state,
    compare_uncond_bc_checkpoints,
    inspect_uncond_bc_checkpoint,
    load_uncond_bc_checkpoint,
    restore_rng_state,
    save_uncond_bc_checkpoint,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(3, 5), nn.GELU(), nn.Linear(5, 3))


class _ActionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.head = nn.Linear(3, 3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = value + block.ffn(value)
        return self.head(value)


class _Scaler:
    def __init__(self) -> None:
        self.value = 1

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = int(state["value"])


def _components(base: _ActionModel | None = None):
    model = copy.deepcopy(base) if base is not None else _ActionModel()
    adapter = inject_action_dit_lora(
        model,
        RegimeLoRAConfig(
            rank=2,
            alpha=2.0,
            dropout=0.0,
            target_groups=(ActionLoRATargetGroup.FFN,),
        ),
    )
    optimizer = torch.optim.AdamW(
        adapter.lora_parameters(),
        lr=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 - 0.01 * step,
    )
    return model, adapter, optimizer, scheduler, _Scaler()


def _update(model, adapter, optimizer, scheduler) -> None:
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.randn(4, 3)
    targets = torch.randn(4, 3)
    with adapter.use_regime(PolicyRegime.UNCOND):
        loss = (model(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()


def _assert_nested_equal(first, second) -> None:
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert set(first) == set(second)
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert type(first) is type(second)
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _assert_nested_equal(left, right)
    else:
        assert first == second


def test_bc_checkpoint_strict_round_trip_and_inspector(tmp_path) -> None:
    torch.manual_seed(3)
    parent = _ActionModel()
    model, adapter, optimizer, scheduler, scaler = _components(parent)
    _update(model, adapter, optimizer, scheduler)
    scaler.value = 7
    contract = {"resolved_config_sha256": "1" * 64, "world_size": 1}
    checkpoint = tmp_path / "step_000001.pt"
    save_uncond_bc_checkpoint(
        checkpoint,
        adapter=adapter,
        parent_checkpoint_sha256="2" * 64,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        grad_scaler=scaler,
        global_step=1,
        epoch=0,
        sampler_offset=11,
        rng_by_rank=[capture_rng_state()],
        contract=contract,
        provenance={"stats_sha256": "3" * 64},
        trainer_state={
            "best_validation_loss_action_bc": 0.25,
            "best_step": 1,
            "epochs_without_improvement": 0,
            "nonzero_update_count": 1,
        },
    )

    report = inspect_uncond_bc_checkpoint(checkpoint)
    assert report["result"] == "PASS"
    assert report["global_step"] == 1
    assert report["sampler_offset"] == 11
    assert report["trainer_state"] == {
        "best_validation_loss_action_bc": 0.25,
        "best_step": 1,
        "epochs_without_improvement": 0,
        "nonzero_update_count": 1,
    }
    assert report["lora_tensor_count"] == 8
    assert report["optimizer_tensor_count"] > 0
    assert report["contains_frozen_fastwam_tensors"] is False
    assert report["contains_gate_tensors"] is False
    assert report["contains_value_head_tensors"] is False
    assert report["contains_raw_training_samples"] is False

    (
        restored_model,
        restored,
        restored_optimizer,
        restored_scheduler,
        restored_scaler,
    ) = _components(parent)
    payload = load_uncond_bc_checkpoint(
        checkpoint,
        adapter=restored,
        expected_parent_checkpoint_sha256="2" * 64,
        expected_contract=contract,
        optimizer=restored_optimizer,
        lr_scheduler=restored_scheduler,
        grad_scaler=restored_scaler,
    )
    del restored_model
    for name, value in adapter.lora_state_dict().items():
        assert torch.equal(restored.lora_state_dict()[name], value)
    _assert_nested_equal(optimizer.state_dict(), restored_optimizer.state_dict())
    _assert_nested_equal(scheduler.state_dict(), restored_scheduler.state_dict())
    assert restored_scaler.value == 7
    assert payload["sampler_offset"] == 11
    assert payload["trainer_state"] == report["trainer_state"]


def test_bc_checkpoint_rejects_parent_and_config_mismatch(tmp_path) -> None:
    model, adapter, optimizer, scheduler, scaler = _components()
    checkpoint = tmp_path / "state.pt"
    contract = {"resolved_config_sha256": "4" * 64}
    save_uncond_bc_checkpoint(
        checkpoint,
        adapter=adapter,
        parent_checkpoint_sha256="5" * 64,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        grad_scaler=scaler,
        global_step=0,
        epoch=0,
        sampler_offset=0,
        rng_by_rank=[capture_rng_state()],
        contract=contract,
        provenance={},
    )

    with pytest.raises(ValueError, match="parent hash mismatch"):
        load_uncond_bc_checkpoint(
            checkpoint,
            adapter=adapter,
            expected_parent_checkpoint_sha256="6" * 64,
            expected_contract=contract,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            grad_scaler=scaler,
        )
    with pytest.raises(ValueError, match="config/data contract mismatch"):
        load_uncond_bc_checkpoint(
            checkpoint,
            adapter=adapter,
            expected_parent_checkpoint_sha256="5" * 64,
            expected_contract={"resolved_config_sha256": "7" * 64},
            optimizer=optimizer,
            lr_scheduler=scheduler,
            grad_scaler=scaler,
        )
    del model


def test_interrupted_resume_matches_lora_optimizer_scheduler_rng_and_next_batch(
    tmp_path,
) -> None:
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    parent = _ActionModel()
    uninterrupted_model, uninterrupted, optimizer, scheduler, scaler = _components(
        parent
    )
    _update(uninterrupted_model, uninterrupted, optimizer, scheduler)
    _update(uninterrupted_model, uninterrupted, optimizer, scheduler)
    checkpoint = tmp_path / "resume.pt"
    contract = {"resolved_config_sha256": "8" * 64}
    save_uncond_bc_checkpoint(
        checkpoint,
        adapter=uninterrupted,
        parent_checkpoint_sha256="9" * 64,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        grad_scaler=scaler,
        global_step=2,
        epoch=1,
        sampler_offset=5,
        rng_by_rank=[capture_rng_state()],
        contract=contract,
        provenance={},
        trainer_state={
            "best_validation_loss_action_bc": 0.5,
            "best_step": 2,
            "epochs_without_improvement": 1,
            "nonzero_update_count": 2,
        },
    )

    expected_next_python = random.random()
    expected_next_numpy = np.random.random()
    expected_next_batch = torch.randn(2, 3)
    restore_rng_state(torch.load(checkpoint, weights_only=False)["rng_by_rank"][0])
    _update(uninterrupted_model, uninterrupted, optimizer, scheduler)
    _update(uninterrupted_model, uninterrupted, optimizer, scheduler)

    resumed_model, resumed, resumed_optimizer, resumed_scheduler, resumed_scaler = (
        _components(parent)
    )
    payload = load_uncond_bc_checkpoint(
        checkpoint,
        adapter=resumed,
        expected_parent_checkpoint_sha256="9" * 64,
        expected_contract=contract,
        optimizer=resumed_optimizer,
        lr_scheduler=resumed_scheduler,
        grad_scaler=resumed_scaler,
    )
    assert payload["trainer_state"] == {
        "best_validation_loss_action_bc": 0.5,
        "best_step": 2,
        "epochs_without_improvement": 1,
        "nonzero_update_count": 2,
    }
    restore_rng_state(payload["rng_by_rank"][0])
    assert random.random() == expected_next_python
    assert np.random.random() == expected_next_numpy
    assert torch.equal(torch.randn(2, 3), expected_next_batch)
    restore_rng_state(payload["rng_by_rank"][0])
    _update(resumed_model, resumed, resumed_optimizer, resumed_scheduler)
    _update(resumed_model, resumed, resumed_optimizer, resumed_scheduler)

    for name, value in uninterrupted.lora_state_dict().items():
        assert torch.equal(resumed.lora_state_dict()[name], value)
    _assert_nested_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
    _assert_nested_equal(scheduler.state_dict(), resumed_scheduler.state_dict())


def test_inspector_fails_closed_on_frozen_or_sample_tensor(tmp_path) -> None:
    model, adapter, optimizer, scheduler, scaler = _components()
    checkpoint = tmp_path / "valid.pt"
    save_uncond_bc_checkpoint(
        checkpoint,
        adapter=adapter,
        parent_checkpoint_sha256="a" * 64,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        grad_scaler=scaler,
        global_step=0,
        epoch=0,
        sampler_offset=0,
        rng_by_rank=[capture_rng_state()],
        contract={"resolved_config_sha256": "b" * 64},
        provenance={},
    )
    payload = torch.load(checkpoint, weights_only=False)
    payload["provenance"]["observation"] = torch.ones(2)
    malformed = tmp_path / "malformed.pt"
    torch.save(payload, malformed)

    with pytest.raises(ValueError, match="outside LoRA/trainer state"):
        inspect_uncond_bc_checkpoint(malformed)
    del model


def test_checkpoint_rejects_inconsistent_trainer_state(tmp_path) -> None:
    model, adapter, optimizer, scheduler, scaler = _components()
    checkpoint = tmp_path / "valid_state.pt"
    contract = {"resolved_config_sha256": "c" * 64}
    save_uncond_bc_checkpoint(
        checkpoint,
        adapter=adapter,
        parent_checkpoint_sha256="d" * 64,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        grad_scaler=scaler,
        global_step=1,
        epoch=0,
        sampler_offset=4,
        rng_by_rank=[capture_rng_state()],
        contract=contract,
        provenance={},
        trainer_state={
            "best_validation_loss_action_bc": 1.0,
            "best_step": 1,
            "epochs_without_improvement": 0,
            "nonzero_update_count": 1,
        },
    )
    malformed = torch.load(checkpoint, weights_only=False)
    malformed["trainer_state"]["nonzero_update_count"] = 2
    malformed_path = tmp_path / "invalid_state.pt"
    torch.save(malformed, malformed_path)

    with pytest.raises(ValueError, match="cannot exceed global_step"):
        inspect_uncond_bc_checkpoint(malformed_path)
    with pytest.raises(ValueError, match="cannot exceed global_step"):
        load_uncond_bc_checkpoint(
            malformed_path,
            adapter=adapter,
            expected_parent_checkpoint_sha256="d" * 64,
            expected_contract=contract,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            grad_scaler=scaler,
        )
    del model


def test_checkpoint_comparator_is_exact_and_excludes_only_provenance(tmp_path) -> None:
    model, adapter, optimizer, scheduler, scaler = _components()
    first = tmp_path / "first.pt"
    save_uncond_bc_checkpoint(
        first,
        adapter=adapter,
        parent_checkpoint_sha256="e" * 64,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        grad_scaler=scaler,
        global_step=0,
        epoch=0,
        sampler_offset=0,
        rng_by_rank=[capture_rng_state()],
        contract={"resolved_config_sha256": "f" * 64},
        provenance={"output_dir": "/first"},
    )
    payload = torch.load(first, weights_only=False)
    payload["provenance"]["output_dir"] = "/second"
    second = tmp_path / "second.pt"
    torch.save(payload, second)

    exact = compare_uncond_bc_checkpoints(first, second)
    assert exact["result"] == "PASS"
    assert exact["exact_training_state"] is True
    assert exact["excluded_paths"] == ["provenance"]

    tensor = next(iter(payload["adapter"]["state_dict"].values()))
    tensor.add_(1)
    changed = tmp_path / "changed.pt"
    torch.save(payload, changed)
    mismatch = compare_uncond_bc_checkpoints(first, changed)
    assert mismatch["result"] == "FAIL"
    assert mismatch["groups"]["adapter"]["mismatch_count"] == 1
    del model
