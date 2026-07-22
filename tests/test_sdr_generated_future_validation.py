import copy
import json

import pytest
import torch

from fastwam.adaptive_gate.sdr_generated_validation import (
    action_distance,
    build_cache_metadata,
    generated_future_sensitivity_gate,
    load_latent_cache,
    no_read_uncond_parity,
    validate_cache_metadata,
    write_latent_cache,
)


def _solver_contract():
    scheduler = {
        "scheduler_class": "test.Scheduler",
        "num_train_timesteps": 1000,
        "configured_shift": 5.0,
        "effective_shift": 5.0,
        "inference_steps": 20,
    }
    return {
        "schema": "fastwam-inference-solver-v1",
        "sigma_shift_override": None,
        "video": scheduler,
        "action": dict(scheduler),
        "branch_semantics": {
            "uncond": "action_only",
            "idm": "video_then_future_conditioned_action",
        },
    }


def _metadata(tmp_path):
    artifacts = {}
    for name in ("e_i.pt", "e_i.yaml", "stats.json"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        artifacts[name] = path
    manifest = tmp_path / "validation.json"
    manifest.write_text(
        json.dumps({"samples": [{"sample_id": "sample-1"}]}),
        encoding="utf-8",
    )
    return build_cache_metadata(
        e_i_checkpoint=artifacts["e_i.pt"],
        e_i_config=artifacts["e_i.yaml"],
        dataset_stats=artifacts["stats.json"],
        validation_manifest=manifest,
        sample_id="sample-1",
        solver_contract=_solver_contract(),
        video_state_sha256="a" * 64,
        proprio_state_sha256="b" * 64,
        seed=20260721,
    )


def test_latent_cache_round_trip_and_exact_provenance(tmp_path):
    metadata = _metadata(tmp_path)
    path = tmp_path / "cache.pt"
    latents = torch.randn(1, 4, 9, 2, 3)

    write_latent_cache(path, video_latents=latents, metadata=metadata)
    loaded = load_latent_cache(path, expected_metadata=metadata)

    torch.testing.assert_close(loaded, latents)


@pytest.mark.parametrize(
    "field",
    [
        "seed",
        "solver_fingerprint",
        "video_state_sha256",
        "proprio_state_sha256",
        "sample_id",
    ],
)
def test_cache_invalidates_on_generation_or_model_drift(tmp_path, field):
    metadata = _metadata(tmp_path)
    changed = copy.deepcopy(metadata)
    changed[field] = "changed" if field != "seed" else metadata["seed"] + 1

    with pytest.raises(ValueError, match="provenance changed"):
        validate_cache_metadata(metadata, changed)


def test_cache_rejects_payload_with_rgb_or_other_unregistered_content(tmp_path):
    metadata = _metadata(tmp_path)
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "metadata": metadata,
            "video_latents": torch.zeros(1, 4, 9, 2, 2),
            "rgb": torch.zeros(3, 33, 16, 16),
        },
        path,
    )

    with pytest.raises(ValueError, match="exactly"):
        load_latent_cache(path, expected_metadata=metadata)


def test_sensitivity_and_no_read_uncond_parity_have_distinct_claims():
    records = [
        {"valid_no_read_normalized_action_l2": value}
        for value in (0.002, 0.003, 0.0001, 0.004)
    ]
    sensitivity = generated_future_sensitivity_gate(records)
    parity = no_read_uncond_parity(
        torch.zeros(32, 7),
        torch.full((32, 7), 5e-5),
    )

    assert sensitivity["pass"]
    assert "not task usefulness" in sensitivity["interpretation"]
    assert parity["pass"]
    assert action_distance(torch.zeros(2, 2), torch.ones(2, 2))["l2"] == 1.0
