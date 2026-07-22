#!/usr/bin/env python3
"""Run E1-P0.5 without optimizer updates on a fixed LIBERO manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from fastwam.adaptive_gate.provenance import (
    inference_solver_contract,
    inference_solver_fingerprint,
)
from fastwam.adaptive_gate.sdr_contracts import (
    PREFLIGHT_DECISION_SCHEMA,
    artifact_record,
    atomic_json,
    read_json,
    validate_e_i_lineage_manifest,
    validate_warmstart_parity_evidence,
)
from fastwam.adaptive_gate.sdr_generated_validation import (
    VALIDATION_SCHEMA,
    action_distance,
    build_cache_metadata,
    generated_future_sensitivity_gate,
    infer_action_from_cached_video_latents,
    load_latent_cache,
    module_state_value_sha256,
    no_read_uncond_parity,
    write_latent_cache,
)
from fastwam.adaptive_gate.sdr_delta import load_action_dit_delta_into_model
from fastwam.adaptive_gate.sdr_preflight import (
    DEFAULT_CANDIDATE_WEIGHTS,
    ExactGradientAccumulator,
    check_loss_arithmetic,
    diagnostic_action_noise_coupling,
    diagnostic_parameter_groups,
    select_preflight_weights,
    weighted_descent_margins,
)
from fastwam.adaptive_gate.sdr_validation import (
    read_manifest,
    validate_validation_manifest,
)
from fastwam.adaptive_gate.warm_start import strict_standalone_idm_warm_start


GRADIENT_SCHEMA = "fastwam-sdr-gradient-diagnostics-v1"
RUN_SCHEMA = "fastwam-sdr-preflight-run-manifest-v1"


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _batch_sample(dataset, sample_index: int) -> dict[str, Any]:
    from torch.utils.data._utils.collate import default_collate

    sample = dataset.get_strict(int(sample_index))
    return default_collate([sample])


def _fixed_shards(samples: Sequence[Mapping[str, Any]], count: int = 8):
    if len(samples) < count or len(samples) % count:
        raise ValueError(
            f"Validation sample count {len(samples)} must divide into {count} shards."
        )
    size = len(samples) // count
    return [list(samples[index * size : (index + 1) * size]) for index in range(count)]


def _seed_all(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _enrich_gradient_statistics(diagnostics: dict[str, Any]) -> None:
    weights = tuple(DEFAULT_CANDIDATE_WEIGHTS) + (0.25,)
    for values in diagnostics["groups"].values():
        values["margins_by_weight"] = {
            str(weight): weighted_descent_margins(values, weight)
            for weight in sorted(set(weights))
        }
    for shard in diagnostics["shards"]:
        for values in shard["groups"].values():
            values["margins_by_weight"] = {
                str(weight): weighted_descent_margins(values, weight)
                for weight in sorted(set(weights))
            }
    summaries = {}
    for group_name in diagnostics["groups"]:
        summaries[group_name] = {}
        for statistic in ("cosine", "idm_norm", "uncond_norm"):
            values = [
                shard["groups"][group_name][statistic]
                for shard in diagnostics["shards"]
                if shard["groups"][group_name][statistic] is not None
            ]
            summaries[group_name][statistic] = (
                {
                    "median": _percentile(values, 0.5),
                    "p10": _percentile(values, 0.1),
                    "p90": _percentile(values, 0.9),
                }
                if values
                else None
            )
        for weight in sorted(set(weights)):
            for objective in ("idm_margin", "uncond_margin"):
                values = [
                    shard["groups"][group_name]["margins_by_weight"][str(weight)][
                        objective
                    ]
                    for shard in diagnostics["shards"]
                ]
                summaries[group_name][f"{objective}@{weight}"] = {
                    "median": _percentile(values, 0.5),
                    "p10": _percentile(values, 0.1),
                    "p90": _percentile(values, 0.9),
                    "negative_fraction": sum(value <= 0.0 for value in values)
                    / len(values),
                }
    diagnostics["shard_margin_summaries"] = summaries


def _run_gradient_replay(
    *,
    model,
    dataset,
    samples: Sequence[Mapping[str, Any]],
    coupling: str,
    repeats: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    accumulator = ExactGradientAccumulator(
        diagnostic_parameter_groups(model, video_final_blocks=1)
    )
    loss_rows = []
    for shard_index, shard_samples in enumerate(_fixed_shards(samples)):
        accumulator.start_shard(shard_index)
        for sample in shard_samples:
            for replay_index in range(int(repeats)):
                replay_seed = (
                    int(seed)
                    + shard_index * 100_000
                    + int(sample["sample_index"]) * 10
                    + replay_index
                )
                _seed_all(replay_seed)
                batch = _batch_sample(dataset, int(sample["sample_index"]))
                model._capture_raw_dual_regime_losses = True
                try:
                    with diagnostic_action_noise_coupling(model, coupling):
                        total, loss_dict = model.training_loss(batch)
                    raw = model._raw_dual_regime_losses
                    idm_raw = raw["idm"]
                    uncond_raw = raw["uncond"]
                    action_scale = float(model.loss_lambda_action)
                    arithmetic = check_loss_arithmetic(
                        idm_raw=action_scale * float(idm_raw.detach().item()),
                        uncond_raw=action_scale
                        * float(uncond_raw.detach().item()),
                        weight=float(loss_dict["action_regime_weight_uncond"]),
                        idm_contribution=float(loss_dict["loss_action_idm"]),
                        uncond_contribution=float(
                            loss_dict["loss_action_uncond"]
                        ),
                        combined=float(loss_dict["loss_action_combined"]),
                        tolerance=2e-6,
                    )
                    accumulator.accumulate(idm_raw, uncond_raw)
                    loss_rows.append(
                        {
                            "coupling": coupling,
                            "shard_index": shard_index,
                            "sample_id": sample["sample_id"],
                            "sample_index": int(sample["sample_index"]),
                            "replay_index": replay_index,
                            "seed": replay_seed,
                            "raw_idm": float(idm_raw.detach().item()),
                            "raw_uncond": float(uncond_raw.detach().item()),
                            "weight": float(
                                loss_dict["action_regime_weight_uncond"]
                            ),
                            "idm_contribution": float(
                                loss_dict["loss_action_idm"]
                            ),
                            "uncond_contribution": float(
                                loss_dict["loss_action_uncond"]
                            ),
                            "combined_action": float(
                                loss_dict["loss_action_combined"]
                            ),
                            "fixed_noise_video_denoising_loss": float(
                                loss_dict["loss_video"]
                            ),
                            "weighted_uncond_over_idm": (
                                float(loss_dict["action_regime_weight_uncond"])
                                * float(uncond_raw.detach().item())
                                / max(float(idm_raw.detach().item()), 1e-12)
                            ),
                            "arithmetic": arithmetic,
                        }
                    )
                finally:
                    model._capture_raw_dual_regime_losses = False
                    if hasattr(model, "_raw_dual_regime_losses"):
                        delattr(model, "_raw_dual_regime_losses")
                    if "total" in locals():
                        del total
                    del batch
            torch.cuda.empty_cache()
        accumulator.finish_shard()
    diagnostics = accumulator.finalize()
    diagnostics["coupling"] = coupling
    diagnostics["replays_per_sample"] = int(repeats)
    diagnostics["replay_count"] = len(loss_rows)
    diagnostics["optimizer_steps"] = 0
    diagnostics["param_grad_mutated"] = any(
        parameter.grad is not None for parameter in accumulator.parameters
    )
    diagnostics["accumulator_storage_bytes_at_finalize"] = (
        accumulator.storage_bytes()
    )
    _enrich_gradient_statistics(diagnostics)
    del accumulator
    torch.cuda.empty_cache()
    return diagnostics, loss_rows


def _denormalize_action(action: torch.Tensor, dataset) -> torch.Tensor:
    processor = dataset.lerobot_dataset.processor
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("Expected one merged LIBERO action key.")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    value = action.detach().to(device="cpu", dtype=torch.float32).unsqueeze(0)
    return normalizer.backward(value)[0]


def _condition_metrics(
    action: torch.Tensor,
    *,
    gt_action: torch.Tensor,
    e_i_action: torch.Tensor,
    dataset,
) -> dict[str, Any]:
    denormalized = _denormalize_action(action, dataset)
    denormalized_gt = _denormalize_action(gt_action, dataset)
    return {
        "normalized_error": action_distance(action, gt_action),
        "denormalized_error": action_distance(
            denormalized,
            denormalized_gt,
        ),
        "e_i_agreement": action_distance(action, e_i_action),
    }


def _aggregate_values(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot aggregate an empty metric sequence.")
    numeric = [float(value) for value in values]
    return {
        "mean": sum(numeric) / len(numeric),
        "median": _percentile(numeric, 0.5),
        "p10": _percentile(numeric, 0.1),
        "p90": _percentile(numeric, 0.9),
        "min": min(numeric),
        "max": max(numeric),
    }


def _generated_action_aggregates(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    condition_names = tuple(records[0]["conditions"])
    conditions = {}
    for condition in condition_names:
        conditions[condition] = {}
        for metric_group in (
            "normalized_error",
            "denormalized_error",
            "e_i_agreement",
        ):
            conditions[condition][metric_group] = {
                metric: _aggregate_values(
                    [
                        row["conditions"][condition][metric_group][metric]
                        for row in records
                    ]
                )
                for metric in ("l1", "l2", "max_abs")
            }
    sensitivity = {
        name: _aggregate_values([float(row[field]) for row in records])
        for name, field in (
            ("valid_no_read", "valid_no_read_normalized_action_l2"),
            ("valid_repeat", "valid_repeat_normalized_action_l2"),
            ("valid_shuffle", "valid_shuffle_normalized_action_l2"),
        )
    }
    return {"conditions": conditions, "sensitivity": sensitivity}


def _generate_future_validation(
    *,
    model,
    dataset,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
    e_i_checkpoint: Path,
    e_i_config: Path,
    dataset_stats: Path,
    output_dir: Path,
    inference_steps: int,
    seed: int,
    sigma_shift: float | None,
    cache_source_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    solver = inference_solver_contract(
        model,
        video_inference_steps=inference_steps,
        action_inference_steps=inference_steps,
        sigma_shift=sigma_shift,
    )
    solver_path = output_dir / "solver_contract.json"
    atomic_json(solver_path, solver)
    video_sha = module_state_value_sha256(model.video_expert)
    proprio_sha = module_state_value_sha256(model.proprio_encoder)
    verified = dict(artifacts)
    verified["validation_manifest"] = artifact_record(manifest_path)
    cache_dir = output_dir / "generated_future_cache"
    source_validation_record = None
    source_records = {}
    if cache_source_dir is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        source_validation_path = (
            cache_source_dir.expanduser().resolve()
            / "generated_future_validation.json"
        )
        source_validation_record = artifact_record(source_validation_path)
        source_validation = read_json(source_validation_path)
        if (
            source_validation.get("schema") != VALIDATION_SCHEMA
            or source_validation.get("status") != "PASS"
            or source_validation.get("cache_created") is not True
            or source_validation.get(
                "direct_generated_cached_valid_parity", {}
            ).get("pass")
            is not True
        ):
            raise ValueError(
                "Generated-future cache source is not a PASS S0 cache creation "
                "with direct/cached generated-IDM parity."
            )
        if source_validation.get("solver_fingerprint") != (
            inference_solver_fingerprint(solver)
        ):
            raise ValueError("Generated-future cache solver contract changed.")
        if (
            source_validation.get("video_state_sha256") != video_sha
            or source_validation.get("proprio_state_sha256") != proprio_sha
        ):
            raise ValueError(
                "Generated-future cache is invalid after video/proprio drift."
            )
        source_records = {
            str(row["sample_id"]): row
            for row in source_validation.get("records", ())
            if isinstance(row, Mapping) and row.get("sample_id")
        }
        if set(source_records) != {
            str(row["sample_id"]) for row in manifest["samples"]
        }:
            raise ValueError(
                "Generated-future cache source does not cover this manifest."
            )
    donors = {
        row["donor_id"]: row for row in manifest["shuffle_donors"]
    }
    donor_cache = {}
    records = []
    parity_results = []
    direct_cache_parity = []
    for ordinal, sample_meta in enumerate(manifest["samples"]):
        sample = _batch_sample(dataset, int(sample_meta["sample_index"]))
        video = sample["video"]
        input_image = video[:, :, 0]
        proprio = sample["proprio"][:, 0]
        context = sample["context"]
        context_mask = sample["context_mask"]
        gt_action = sample["action"][0].detach().float().cpu()
        action_horizon = int(gt_action.shape[0])
        num_video_frames = int(video.shape[2])
        action_seed = int(seed) + ordinal
        cache_metadata = build_cache_metadata(
            e_i_checkpoint=e_i_checkpoint,
            e_i_config=e_i_config,
            dataset_stats=dataset_stats,
            validation_manifest=manifest_path,
            sample_id=sample_meta["sample_id"],
            solver_contract=solver,
            video_state_sha256=video_sha,
            proprio_state_sha256=proprio_sha,
            seed=action_seed,
            verified_artifacts=verified,
        )
        source_record = source_records.get(str(sample_meta["sample_id"]))
        if source_record is None:
            generated = model.infer_action(
                prompt=None,
                input_image=input_image,
                action_horizon=action_horizon,
                num_video_frames=num_video_frames,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=inference_steps,
                sigma_shift=sigma_shift,
                seed=action_seed,
                rand_device="cpu",
                force_branch="idm",
                return_video_latents=True,
            )
            generated_latents = generated["video_latents"]
            e_i_reference = generated["action"].detach().float().cpu()
            cache_path = cache_dir / f"{sample_meta['sample_id']}.pt"
            write_latent_cache(
                cache_path,
                video_latents=generated_latents,
                metadata=cache_metadata,
            )
        else:
            if int(source_record.get("action_seed", -1)) != action_seed:
                raise ValueError("S0 action reference seed changed.")
            cache_record = source_record.get("recipient_cache")
            if not isinstance(cache_record, Mapping):
                raise ValueError("S0 recipient cache record is missing.")
            cache_path = Path(str(cache_record.get("path", ""))).resolve()
            actual_cache_record = artifact_record(cache_path)
            if cache_record.get("sha256") != actual_cache_record["sha256"]:
                raise ValueError("S0 recipient latent cache changed.")
            generated_latents = load_latent_cache(
                cache_path,
                expected_metadata=cache_metadata,
            )
            raw_reference = source_record.get("standalone_e_i_reference_action")
            e_i_reference = torch.tensor(raw_reference, dtype=torch.float32)
            if tuple(e_i_reference.shape) != tuple(gt_action.shape):
                raise ValueError("S0 E-I reference action shape changed.")

        donor_meta = donors[sample_meta["shuffle_donor_id"]]
        donor_id = donor_meta["donor_id"]
        if donor_id not in donor_cache:
            donor_seed = (
                int(seed) + 1_000_000 + ordinal
                if source_record is None
                else int(source_record.get("shuffle_donor_seed", -1))
            )
            if donor_seed < 0:
                raise ValueError("S0 shuffle donor seed is missing.")
            donor_metadata = build_cache_metadata(
                e_i_checkpoint=e_i_checkpoint,
                e_i_config=e_i_config,
                dataset_stats=dataset_stats,
                validation_manifest=manifest_path,
                sample_id=donor_id,
                solver_contract=solver,
                video_state_sha256=video_sha,
                proprio_state_sha256=proprio_sha,
                seed=donor_seed,
                verified_artifacts=verified,
            )
            if source_record is None:
                donor_sample = _batch_sample(
                    dataset,
                    int(donor_meta["sample_index"]),
                )
                donor_result = model.infer_action(
                    prompt=None,
                    input_image=donor_sample["video"][:, :, 0],
                    action_horizon=int(donor_sample["action"].shape[1]),
                    num_video_frames=int(donor_sample["video"].shape[2]),
                    proprio=donor_sample["proprio"][:, 0],
                    context=donor_sample["context"],
                    context_mask=donor_sample["context_mask"],
                    num_inference_steps=inference_steps,
                    sigma_shift=sigma_shift,
                    seed=donor_seed,
                    rand_device="cpu",
                    force_branch="idm",
                    return_video_latents=True,
                )
                donor_latents = donor_result["video_latents"]
                donor_path = cache_dir / f"{donor_id}.pt"
                write_latent_cache(
                    donor_path,
                    video_latents=donor_latents,
                    metadata=donor_metadata,
                )
                del donor_sample, donor_result
            else:
                donor_record = source_record.get("shuffle_donor_cache")
                if not isinstance(donor_record, Mapping):
                    raise ValueError("S0 shuffle donor cache record is missing.")
                donor_path = Path(str(donor_record.get("path", ""))).resolve()
                actual_donor_record = artifact_record(donor_path)
                if donor_record.get("sha256") != actual_donor_record["sha256"]:
                    raise ValueError("S0 shuffle donor latent cache changed.")
                donor_latents = load_latent_cache(
                    donor_path,
                    expected_metadata=donor_metadata,
                )
            donor_cache[donor_id] = (
                donor_latents,
                donor_path,
                donor_metadata,
                donor_seed,
            )
        donor_latents, donor_path, donor_metadata, donor_seed = donor_cache[
            donor_id
        ]
        donor_match = {
            "suite": donor_meta["suite"],
            "task_index": int(donor_meta["task_index"]),
        }

        common = {
            "model": model,
            "action_horizon": action_horizon,
            "context": context,
            "context_mask": context_mask,
            "proprio": proprio,
            "seed": action_seed,
            "action_inference_steps": inference_steps,
            "sigma_shift": sigma_shift,
            "rand_device": "cpu",
        }
        valid = infer_action_from_cached_video_latents(
            video_latents=generated_latents,
            control="valid_idm",
            **common,
        )
        no_read = infer_action_from_cached_video_latents(
            video_latents=generated_latents,
            control="no_read",
            **common,
        )
        repeat = infer_action_from_cached_video_latents(
            video_latents=generated_latents,
            control="repeat_current",
            **common,
        )
        shuffled = infer_action_from_cached_video_latents(
            video_latents=generated_latents,
            control="shuffled",
            shuffled_future_latents=donor_latents,
            shuffled_future_metadata=donor_match,
            expected_donor_metadata=donor_match,
            **common,
        )
        gt_latents = model._encode_video_latents(
            video.to(device=model.device, dtype=model.torch_dtype),
            tiled=False,
        ).detach().cpu()
        teacher_forced = infer_action_from_cached_video_latents(
            video_latents=gt_latents,
            control="valid_idm",
            **common,
        )
        uncond = model.infer_action(
            prompt=None,
            input_image=input_image,
            action_horizon=action_horizon,
            num_video_frames=num_video_frames,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=inference_steps,
            sigma_shift=sigma_shift,
            seed=action_seed,
            rand_device="cpu",
            force_branch="base",
        )["action"]
        parity = no_read_uncond_parity(no_read, uncond, tolerance=1e-4)
        parity_results.append(parity)
        cache_parity = (
            action_distance(valid, e_i_reference)
            if source_record is None
            else None
        )
        if cache_parity is not None:
            direct_cache_parity.append(cache_parity)
        record = {
            "sample_id": sample_meta["sample_id"],
            "sample_index": int(sample_meta["sample_index"]),
            "suite": sample_meta["suite"],
            "task_index": int(sample_meta["task_index"]),
            "action_seed": action_seed,
            "standalone_e_i_reference_action": e_i_reference.tolist(),
            "recipient_cache": artifact_record(cache_path),
            "recipient_cache_key": cache_metadata["cache_key"],
            "shuffle_donor_id": donor_id,
            "shuffle_donor_seed": donor_seed,
            "shuffle_donor_cache": artifact_record(donor_path),
            "shuffle_donor_cache_key": donor_metadata["cache_key"],
            "conditions": {
                "gt_teacher_forced_future": _condition_metrics(
                    teacher_forced,
                    gt_action=gt_action,
                    e_i_action=e_i_reference,
                    dataset=dataset,
                ),
                "valid_self_generated_future": _condition_metrics(
                    valid,
                    gt_action=gt_action,
                    e_i_action=e_i_reference,
                    dataset=dataset,
                ),
                "no_read": _condition_metrics(
                    no_read,
                    gt_action=gt_action,
                    e_i_action=e_i_reference,
                    dataset=dataset,
                ),
                "repeat_current": _condition_metrics(
                    repeat,
                    gt_action=gt_action,
                    e_i_action=e_i_reference,
                    dataset=dataset,
                ),
                "matched_shuffled_future": _condition_metrics(
                    shuffled,
                    gt_action=gt_action,
                    e_i_action=e_i_reference,
                    dataset=dataset,
                ),
                "forced_uncond": _condition_metrics(
                    uncond,
                    gt_action=gt_action,
                    e_i_action=e_i_reference,
                    dataset=dataset,
                ),
                "standalone_e_i_reference": _condition_metrics(
                    e_i_reference,
                    gt_action=gt_action,
                    e_i_action=e_i_reference,
                    dataset=dataset,
                ),
            },
            "no_read_uncond_parity": parity,
            "direct_generated_cached_valid_parity": cache_parity,
            "valid_no_read_normalized_action_l2": action_distance(
                valid,
                no_read,
            )["l2"],
            "valid_repeat_normalized_action_l2": action_distance(
                valid,
                repeat,
            )["l2"],
            "valid_shuffle_normalized_action_l2": action_distance(
                valid,
                shuffled,
            )["l2"],
        }
        records.append(record)
        del sample, generated_latents, gt_latents
        if "generated" in locals():
            del generated
        torch.cuda.empty_cache()

    sensitivity = generated_future_sensitivity_gate(records)
    result = {
        "schema": VALIDATION_SCHEMA,
        "status": "PASS",
        "sample_count": len(records),
        "solver_contract": solver,
        "solver_fingerprint": inference_solver_fingerprint(solver),
        "video_state_sha256": video_sha,
        "proprio_state_sha256": proprio_sha,
        "cache_contents": "video_latents_only",
        "cache_created": cache_source_dir is None,
        "cache_source": source_validation_record,
        "artifact_bindings": {
            name: dict(record)
            for name, record in verified.items()
        },
        "records": records,
        "aggregate": _generated_action_aggregates(records),
        "sensitivity_gate": sensitivity,
        "no_read_uncond_parity": {
            "pass": all(row["pass"] for row in parity_results),
            "max_abs": max(row["max_abs"] for row in parity_results),
            "tolerance": 1e-4,
        },
        "direct_generated_cached_valid_parity": (
            {
                "pass": all(
                    row["max_abs"] <= 1e-4 for row in direct_cache_parity
                ),
                "max_abs": max(
                    row["max_abs"] for row in direct_cache_parity
                ),
                "tolerance": 1e-4,
                "evaluated": True,
            }
            if direct_cache_parity
            else {
                "pass": None,
                "max_abs": None,
                "tolerance": 1e-4,
                "evaluated": False,
                "reason": (
                    "Checkpoint action validation reused the exact S0 latent "
                    "cache and did not rerun video generation."
                ),
            }
        ),
        "interpretation": (
            "Action sensitivity is conditioning-read evidence, not task "
            "usefulness evidence."
        ),
    }
    return result, artifact_record(solver_path)


def _git_commit_record(repo: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if dirty:
        raise ValueError("S-DR preflight requires a clean committed FastWAM tree.")
    return {
        "path": f"git:{commit}",
        "sha256": hashlib.sha256(commit.encode("ascii")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--e-i-checkpoint", required=True)
    parser.add_argument("--e-i-config", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--warmstart-decision", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-delta")
    parser.add_argument("--generated-future-cache-source")
    parser.add_argument("--replays-per-mode", type=int, default=2)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--sigma-shift", type=float)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--no-fail-exit", action="store_true")
    args = parser.parse_args()

    if args.replays_per_mode < 2:
        parser.error("--replays-per-mode must be at least 2")
    if not torch.cuda.is_available():
        parser.error("E1-P0.5 requires CUDA")

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from fastwam.utils import misc

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(output_dir)
    repo = Path(__file__).resolve().parents[1]
    code_record = _git_commit_record(repo)
    config_path = Path(args.resolved_config).expanduser().resolve()
    e_i_checkpoint = Path(args.e_i_checkpoint).expanduser().resolve()
    e_i_config = Path(args.e_i_config).expanduser().resolve()
    stats_path = Path(args.dataset_stats).expanduser().resolve()
    manifest_path = Path(args.validation_manifest).expanduser().resolve()
    lineage_path = Path(args.lineage_manifest).expanduser().resolve()
    warmstart_path = Path(args.warmstart_decision).expanduser().resolve()
    cfg = OmegaConf.load(config_path)
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
    lineage = read_json(lineage_path)
    lineage_artifacts = validate_e_i_lineage_manifest(
        lineage,
        expected_paths={
            "base_model_manifest": lineage["artifacts"][
                "base_model_manifest"
            ]["path"],
            "e_i_checkpoint": e_i_checkpoint,
            "e_i_config": e_i_config,
            "dataset_stats": stats_path,
        },
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = instantiate(
        cfg.model,
        model_dtype=torch.bfloat16,
        device="cuda",
    )
    warm_start = {
        "kind": "standalone_idm",
        "checkpoint": str(e_i_checkpoint),
        "expected_checkpoint_sha256": lineage_artifacts["e_i_checkpoint"][
            "sha256"
        ],
        "source_task": str(cfg.warm_start.source_task),
        "source_config": str(e_i_config),
        "source_dataset_stats": str(stats_path),
    }
    strict_standalone_idm_warm_start(
        model,
        warm_start,
        target_model_config=cfg.model,
        target_dataset_stats=stats_path,
    )
    locked_solver = inference_solver_contract(
        model,
        video_inference_steps=args.inference_steps,
        action_inference_steps=args.inference_steps,
        sigma_shift=args.sigma_shift,
    )
    warmstart_evidence = validate_warmstart_parity_evidence(
        warmstart_path,
        expected_artifacts=lineage_artifacts,
        expected_solver_fingerprint=inference_solver_fingerprint(locked_solver),
        expected_inference_steps=args.inference_steps,
    )
    delta_record = None
    if args.action_delta:
        delta_record = load_action_dit_delta_into_model(
            model,
            delta_checkpoint=args.action_delta,
            parent_checkpoint_sha256=lineage_artifacts["e_i_checkpoint"][
                "sha256"
            ],
        )
    model.eval()
    model.requires_grad_(False)
    model.action_expert.requires_grad_(True)
    model.proprio_encoder.requires_grad_(True)
    for block in model.video_expert.blocks[-1:]:
        block.requires_grad_(True)
    model.action_regime_weight_uncond = 0.05

    dataset = instantiate(
        cfg.data.train,
        pretrained_norm_stats=str(stats_path),
        is_training_set=False,
        val_set_proportion=0.0,
    )
    if len(manifest["samples"]) < 32 or len(manifest["samples"]) > 64:
        raise ValueError("Preflight validation manifest must contain 32-64 samples.")

    common, common_losses = _run_gradient_replay(
        model=model,
        dataset=dataset,
        samples=manifest["samples"],
        coupling="common",
        repeats=args.replays_per_mode,
        seed=args.seed,
    )
    weights = select_preflight_weights(common)
    independent, independent_losses = _run_gradient_replay(
        model=model,
        dataset=dataset,
        samples=manifest["samples"],
        coupling="independent",
        repeats=args.replays_per_mode,
        seed=args.seed,
    )
    gradient_result = {
        "schema": GRADIENT_SCHEMA,
        "status": "PASS",
        "optimizer_steps": 0,
        "manifest_sample_count": len(manifest["samples"]),
        "common": common,
        "independent": independent,
        "loss_records": common_losses + independent_losses,
        "weight_selection": weights,
    }
    atomic_json(output_dir / "gradient_diagnostics.json", gradient_result)

    model.requires_grad_(False)
    generated, solver_record = _generate_future_validation(
        model=model,
        dataset=dataset,
        manifest=manifest,
        manifest_path=manifest_path,
        artifacts=lineage_artifacts,
        e_i_checkpoint=e_i_checkpoint,
        e_i_config=e_i_config,
        dataset_stats=stats_path,
        output_dir=output_dir,
        inference_steps=args.inference_steps,
        seed=args.seed,
        sigma_shift=args.sigma_shift,
        cache_source_dir=(
            Path(args.generated_future_cache_source)
            if args.generated_future_cache_source
            else None
        ),
    )
    fixed_video_losses = [
        float(row["fixed_noise_video_denoising_loss"])
        for row in common_losses
    ]
    generated["fixed_noise_video_denoising_loss"] = {
        "count": len(fixed_video_losses),
        "mean": sum(fixed_video_losses) / len(fixed_video_losses),
        "min": min(fixed_video_losses),
        "max": max(fixed_video_losses),
        "coupling": "common",
    }
    atomic_json(output_dir / "generated_future_validation.json", generated)

    artifacts = {
        **lineage_artifacts,
        "validation_manifest": artifact_record(manifest_path),
        "lineage_manifest": artifact_record(lineage_path),
        "warmstart_decision": warmstart_evidence["decision"],
        "warmstart_parity_result": warmstart_evidence["parity_result"],
        "resolved_config": artifact_record(config_path),
        "solver_contract": solver_record,
        "code_commit": code_record,
    }
    if delta_record is not None:
        artifacts["action_dit_delta"] = artifact_record(args.action_delta)
    failures = []
    if not all(
        row["arithmetic"]["pass"]
        for row in gradient_result["loss_records"]
    ):
        failures.append("loss arithmetic mismatch")
    if common["param_grad_mutated"] or independent["param_grad_mutated"]:
        failures.append("diagnostic replay mutated param.grad")
    if not weights["go"]:
        failures.append("no safe w_cap >= 0.2")
    if not generated["no_read_uncond_parity"]["pass"]:
        failures.append("no-read/forced-UNCOND parity failed")
    if (
        args.generated_future_cache_source is None
        and not generated["direct_generated_cached_valid_parity"]["pass"]
    ):
        failures.append("direct/cached generated-IDM parity failed")
    if not generated["sensitivity_gate"]["pass"]:
        failures.append("S0 has no detectable generated-future sensitivity")
    if not _finite_tree(gradient_result) or not _finite_tree(generated):
        failures.append("non-finite metric")
    if weights["w0"] > 0.05:
        failures.append("w0 exceeds 0.05")
    coupling_summary = {}
    for coupling, diagnostics, rows in (
        ("common", common, common_losses),
        ("independent", independent, independent_losses),
    ):
        action_all = diagnostics["groups"]["action_all"]
        coupling_summary[coupling] = {
            "sample_count": diagnostics["sample_count"],
            "replay_count": diagnostics["replay_count"],
            "action_all": {
                key: action_all[key]
                for key in (
                    "dot",
                    "idm_norm",
                    "uncond_norm",
                    "cosine",
                )
            },
            "action_all_shard_margins": diagnostics[
                "shard_margin_summaries"
            ]["action_all"],
            "final_block_shard_margins": diagnostics[
                "shard_margin_summaries"
            ]["action_blocks_final"],
            "raw_loss": {
                "idm": _aggregate_values(
                    [float(row["raw_idm"]) for row in rows]
                ),
                "uncond": _aggregate_values(
                    [float(row["raw_uncond"]) for row in rows]
                ),
                "weighted_uncond_over_idm": _aggregate_values(
                    [
                        float(row["weighted_uncond_over_idm"])
                        for row in rows
                    ]
                ),
            },
        }
    decision = {
        "schema": PREFLIGHT_DECISION_SCHEMA,
        "status": "PASS" if not failures else "FAIL-DIAGNOSED",
        "plus_full_used": False,
        "artifact_bindings": artifacts,
        "w0": weights["w0"],
        "w_cap": weights["w_cap"],
        "safe_candidate_weights": weights["safe_candidate_weights"],
        "allowed_schedule": weights["schedule"],
        "uncond_gradient_imbalance": weights["w0"] < 0.001,
        "gradient_summary": coupling_summary,
        "warmstart_idm_parity": "PASS",
        "no_read_uncond_parity": generated[
            "no_read_uncond_parity"
        ],
        "generated_future_sensitivity": generated["sensitivity_gate"],
        "failure_conditions": failures,
        "next_stage_authorized": not failures,
        "optimizer_steps": 0,
        "production_shape": {
            "raw_sequence_frames": 33,
            "sampled_video_frames": int(
                _batch_sample(dataset, manifest["samples"][0]["sample_index"])[
                    "video"
                ].shape[2]
            ),
            "height": 224,
            "width": 448,
            "action_shape": [32, 7],
        },
        "no_plus_full_statement": (
            "No Plus-Full outcome was loaded or used for this decision."
        ),
    }
    atomic_json(output_dir / "preflight_decision.json", decision)
    elapsed = time.monotonic() - started
    usage = shutil.disk_usage(output_dir)
    run_manifest = {
        "schema": RUN_SCHEMA,
        "status": decision["status"],
        "plus_full_used": False,
        "artifacts": {
            **artifacts,
            "gradient_diagnostics": artifact_record(
                output_dir / "gradient_diagnostics.json"
            ),
            "generated_future_validation": artifact_record(
                output_dir / "generated_future_validation.json"
            ),
            "preflight_decision": artifact_record(
                output_dir / "preflight_decision.json"
            ),
        },
        "hardware": {
            "hostname": platform.node(),
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "elapsed_seconds": elapsed,
        },
        "disk": {
            "free_bytes": usage.free,
            "total_bytes": usage.total,
        },
        "optimizer_steps": 0,
    }
    atomic_json(output_dir / "run_manifest.json", run_manifest)
    print(
        json.dumps(
            {
                "status": decision["status"],
                "decision": str(output_dir / "preflight_decision.json"),
                "peak_memory_bytes": run_manifest["hardware"][
                    "peak_memory_bytes"
                ],
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        )
    )
    if failures and not args.no_fail_exit:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
