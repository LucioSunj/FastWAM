"""Fail-closed artifact and stage contracts for the S-DR experiment."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .provenance import sha256_file


E_I_LINEAGE_SCHEMA = "fastwam-ei-lineage-v2"
ORIGINAL_WAN_BASE_SCHEMA = "fastwam-original-wan-base-v1"
PREFLIGHT_DECISION_SCHEMA = "fastwam-sdr-preflight-decision-v1"
LEARNING_PROBE_DECISION_SCHEMA = "fastwam-sdr-learning-probe-decision-v1"
FORMAL_TRAINING_DECISION_SCHEMA = "fastwam-sdr-formal-training-decision-v1"

E_I_ARTIFACT_NAMES = (
    "base_model_manifest",
    "e_i_checkpoint",
    "e_i_config",
    "dataset_stats",
)
ORIGINAL_WAN_COMPONENT_NAMES = (
    "wan_video_dit_index",
    "wan_video_dit_shard_1",
    "wan_video_dit_shard_2",
    "wan_video_dit_shard_3",
    "wan_video_vae",
    "action_dit_initial_checkpoint",
)


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}.")
    return payload


def atomic_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def artifact_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _validated_record(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Lineage artifact {name!r} must be an object.")
    path = Path(str(value.get("path", ""))).expanduser().resolve()
    expected_sha = value.get("sha256")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = sha256_file(path)
    if expected_sha != actual_sha:
        raise ValueError(
            f"Lineage artifact {name!r} changed: expected={expected_sha}, "
            f"actual={actual_sha}, path={path}."
        )
    return {
        "path": str(path),
        "sha256": actual_sha,
        "size_bytes": path.stat().st_size,
    }


def _validate_original_wan_base_manifest(
    manifest_path: str | os.PathLike[str],
) -> dict[str, dict[str, Any]]:
    manifest = read_json(manifest_path)
    if manifest.get("schema") != ORIGINAL_WAN_BASE_SCHEMA:
        raise ValueError(
            "Unsupported original-Wan base schema: "
            f"{manifest.get('schema')!r}."
        )
    if manifest.get("status") != "PASS":
        raise ValueError("Original-Wan base manifest is not PASS.")
    if manifest.get("model_id") != "Wan-AI/Wan2.2-TI2V-5B":
        raise ValueError("Original-Wan base manifest has the wrong model_id.")
    if manifest.get("plus_full_used") is not False:
        raise ValueError("Original-Wan base manifest must state plus_full_used=false.")
    raw_components = manifest.get("artifacts")
    if not isinstance(raw_components, Mapping):
        raise ValueError("Original-Wan base manifest has no artifacts mapping.")
    if set(raw_components) != set(ORIGINAL_WAN_COMPONENT_NAMES):
        raise ValueError(
            "Original-Wan base component names must be exactly "
            f"{list(ORIGINAL_WAN_COMPONENT_NAMES)}, got {sorted(raw_components)}."
        )
    return {
        name: _validated_record(name, raw_components[name])
        for name in ORIGINAL_WAN_COMPONENT_NAMES
    }


def validate_e_i_lineage_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_paths: Mapping[str, str | os.PathLike[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify the user-confirmed original-Wan to LIBERO E-I artifact chain."""
    if manifest.get("schema") != E_I_LINEAGE_SCHEMA:
        raise ValueError(
            f"Unsupported E-I lineage schema: {manifest.get('schema')!r}."
        )
    if manifest.get("status") != "PASS":
        raise ValueError("E-I lineage manifest is not PASS.")
    if manifest.get("config_origin") not in {
        "training_emitted",
        "user_provided_training_config",
    }:
        raise ValueError(
            "E-I lineage requires a training-emitted or directly user-provided "
            "training config; a reconstructed config is not admissible."
        )
    if manifest.get("lineage_assertion_origin") not in {
        "training_emitted_manifest",
        "user_attested",
    }:
        raise ValueError("E-I lineage assertion origin is not admissible.")
    if manifest.get("plus_full_used") is not False:
        raise ValueError("E-I lineage manifest must state plus_full_used=false.")

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, Mapping):
        raise ValueError("E-I lineage manifest has no artifacts mapping.")
    artifacts = {
        name: _validated_record(name, raw_artifacts.get(name))
        for name in E_I_ARTIFACT_NAMES
    }
    if expected_paths is not None:
        for name, raw_expected in expected_paths.items():
            if name not in artifacts:
                raise ValueError(f"Unexpected E-I lineage path binding: {name}.")
            expected = Path(raw_expected).expanduser().resolve()
            if expected != Path(artifacts[name]["path"]):
                raise ValueError(
                    f"E-I lineage path mismatch for {name}: "
                    f"manifest={artifacts[name]['path']}, expected={expected}."
                )

    training = manifest.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("E-I lineage manifest has no training relation.")
    base_components = _validate_original_wan_base_manifest(
        artifacts["base_model_manifest"]["path"]
    )
    required_relations = {
        "initializer_kind": "original_wan2.2_ti2v_5b",
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "parent_manifest_sha256": artifacts["base_model_manifest"]["sha256"],
        "output_checkpoint_sha256": artifacts["e_i_checkpoint"]["sha256"],
        "output_config_sha256": artifacts["e_i_config"]["sha256"],
        "dataset_stats_sha256": artifacts["dataset_stats"]["sha256"],
    }
    mismatches = {
        key: (training.get(key), expected)
        for key, expected in required_relations.items()
        if training.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "E-I lineage parent/output relation is inconsistent "
            f"(observed, expected): {mismatches}."
        )
    completed_step = training.get("completed_step")
    if (
        isinstance(completed_step, bool)
        or not isinstance(completed_step, int)
        or completed_step <= 0
    ):
        raise ValueError("E-I lineage completed_step must be a positive integer.")
    task = str(training.get("task", "")).lower()
    if "libero" not in task or "idm" not in task:
        raise ValueError("E-I lineage task must identify LIBERO FastWAM-IDM.")

    import yaml

    config = yaml.safe_load(
        Path(artifacts["e_i_config"]["path"]).read_text(encoding="utf-8")
    )
    model = config.get("model") if isinstance(config, Mapping) else None
    if not isinstance(model, Mapping):
        raise ValueError("E-I config has no model mapping.")
    config_requirements = {
        "_target_": "fastwam.runtime.create_fastwam_idm",
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "skip_dit_load_from_pretrain": False,
        "load_text_encoder": False,
    }
    config_mismatches = {
        key: (model.get(key), expected)
        for key, expected in config_requirements.items()
        if model.get(key) != expected
    }
    if config_mismatches:
        raise ValueError(
            "E-I config does not describe the original-Wan IDM initializer: "
            f"{config_mismatches}."
        )
    action_initializer = Path(
        str(model.get("action_dit_pretrained_path", ""))
    ).name
    expected_action_initializer = Path(
        base_components["action_dit_initial_checkpoint"]["path"]
    ).name
    if action_initializer != expected_action_initializer:
        raise ValueError(
            "E-I config ActionDiT initializer differs from the base manifest: "
            f"{action_initializer!r} != {expected_action_initializer!r}."
        )

    stats = read_json(artifacts["dataset_stats"]["path"])
    dataset_relations = {
        "num_episodes": stats.get("num_episodes"),
        "num_transition": stats.get("num_transition"),
    }
    dataset_mismatches = {
        key: (training.get(key), expected)
        for key, expected in dataset_relations.items()
        if training.get(key) != expected
    }
    if dataset_mismatches:
        raise ValueError(
            "E-I lineage dataset relation is inconsistent: "
            f"{dataset_mismatches}."
        )
    return artifacts


def validate_warmstart_parity_evidence(
    decision_path: str | os.PathLike[str],
    *,
    expected_artifacts: Mapping[str, Mapping[str, Any]],
    expected_solver_fingerprint: str,
    expected_inference_steps: int = 20,
) -> dict[str, Any]:
    """Bind an existing S0 parity decision to this exact E-I lineage."""
    decision_file = Path(decision_path).expanduser().resolve()
    decision = read_json(decision_file)
    if decision.get("status") != "PASS":
        raise ValueError("Warm-start IDM parity decision is not PASS.")
    metrics = decision.get("metrics")
    evidence = metrics.get("evidence") if isinstance(metrics, Mapping) else None
    if not isinstance(evidence, list) or len(evidence) != 1:
        raise ValueError(
            "Warm-start parity decision must bind exactly one parity result."
        )
    record = evidence[0]
    if not isinstance(record, Mapping):
        raise ValueError("Warm-start parity evidence record is malformed.")
    parity_path = Path(str(record.get("path", ""))).expanduser().resolve()
    parity_record = artifact_record(parity_path)
    if record.get("sha256") != parity_record["sha256"]:
        raise ValueError("Warm-start parity result changed after its decision.")
    parity = read_json(parity_path)
    if (
        parity.get("schema") != "fastwam-warmstart-parity-v1"
        or parity.get("status") != "PASS"
        or parity.get("kind") != "standalone_idm_to_s0_fixed_seed_parity"
    ):
        raise ValueError("Warm-start parity result is not an accepted S0 check.")

    observed_artifacts = parity.get("artifacts")
    if not isinstance(observed_artifacts, Mapping):
        raise ValueError("Warm-start parity result has no artifacts mapping.")
    aliases = {
        "e_i_checkpoint": "e_i_checkpoint",
        "e_i_config": "e_i_config",
        "e_i_stats": "dataset_stats",
    }
    mismatches = {}
    for observed_name, expected_name in aliases.items():
        observed = observed_artifacts.get(observed_name)
        expected = expected_artifacts.get(expected_name)
        if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
            mismatches[observed_name] = (observed, expected)
            continue
        if observed.get("sha256") != expected.get("sha256"):
            mismatches[observed_name] = (
                observed.get("sha256"),
                expected.get("sha256"),
            )
    if mismatches:
        raise ValueError(
            "Warm-start parity evidence belongs to different E-I artifacts: "
            f"{mismatches}."
        )
    if int(parity.get("inference_steps", -1)) != int(expected_inference_steps):
        raise ValueError("Warm-start parity used a different solver step count.")
    if parity.get("solver_contract_sha256") != expected_solver_fingerprint:
        raise ValueError("Warm-start parity solver contract changed.")
    actions = parity.get("actions")
    if not isinstance(actions, Mapping) or actions.get("shape") != [32, 7]:
        raise ValueError("Warm-start parity did not validate action shape [32, 7].")
    comparison = parity.get("comparison")
    if (
        not isinstance(comparison, Mapping)
        or float(comparison.get("worst_normalized_error", float("inf"))) > 1.0
    ):
        raise ValueError("Warm-start parity comparison is outside its tolerance.")
    return {
        "decision": artifact_record(decision_file),
        "parity_result": parity_record,
        "solver_fingerprint": expected_solver_fingerprint,
    }


def audit_e_i_lineage_inputs(
    *,
    base_model_manifest: str | os.PathLike[str] | None,
    e_i_checkpoint: str | os.PathLike[str] | None,
    e_i_config: str | os.PathLike[str] | None,
    dataset_stats: str | os.PathLike[str] | None,
    lineage_manifest: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    """Return PASS or NOT-RUN without inferring missing provenance."""
    supplied = {
        "base_model_manifest": base_model_manifest,
        "e_i_checkpoint": e_i_checkpoint,
        "e_i_config": e_i_config,
        "dataset_stats": dataset_stats,
    }
    blockers = []
    artifacts = {}
    for name, raw_path in supplied.items():
        if raw_path is None:
            blockers.append(f"missing required path: {name}")
            continue
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            blockers.append(f"missing required file: {name}={path}")
            continue
        artifacts[name] = artifact_record(path)

    manifest_payload = None
    manifest_record = None
    if lineage_manifest is None:
        blockers.append("missing required path: e_i_lineage_manifest")
    else:
        path = Path(lineage_manifest).expanduser().resolve()
        if not path.is_file():
            blockers.append(f"missing required file: e_i_lineage_manifest={path}")
        else:
            manifest_record = artifact_record(path)
            try:
                manifest_payload = read_json(path)
                if not blockers:
                    validate_e_i_lineage_manifest(
                        manifest_payload,
                        expected_paths={
                            name: record["path"] for name, record in artifacts.items()
                        },
                    )
            except Exception as exc:
                blockers.append(f"invalid E-I lineage manifest: {exc}")

    status = "PASS" if not blockers else "NOT-RUN"
    return {
        "schema": "fastwam-ei-lineage-audit-v1",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "lineage_manifest": manifest_record,
        "lineage_manifest_schema": (
            manifest_payload.get("schema")
            if isinstance(manifest_payload, Mapping)
            else None
        ),
        "blockers": blockers,
        "claim": (
            "Verified exact artifact bindings for the user-attested original-Wan "
            "base -> LIBERO FastWAM-IDM E-I lineage."
            if status == "PASS"
            else "No training update is authorized by this audit."
        ),
    }


def _artifact_sha_map(value: object, *, source: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{source} has no artifact_bindings mapping.")
    output = {}
    for name, record in value.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"{source} artifact {name!r} must be an object.")
        sha = record.get("sha256")
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(char not in "0123456789abcdef" for char in sha)
        ):
            raise ValueError(f"{source} artifact {name!r} has an invalid SHA256.")
        output[str(name)] = sha
    return output


def validate_stage_decision(
    decision: Mapping[str, Any],
    *,
    expected_schema: str,
    expected_bindings: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if decision.get("schema") != expected_schema:
        raise ValueError(
            f"Expected {expected_schema!r}, got {decision.get('schema')!r}."
        )
    if decision.get("status") != "PASS":
        raise ValueError(f"{expected_schema} is not PASS.")
    if decision.get("plus_full_used") is not False:
        raise ValueError(f"{expected_schema} must state plus_full_used=false.")
    bindings = _artifact_sha_map(
        decision.get("artifact_bindings"), source=expected_schema
    )
    if expected_bindings is not None:
        mismatches = {
            name: (bindings.get(name), expected)
            for name, expected in expected_bindings.items()
            if bindings.get(name) != expected
        }
        if mismatches:
            raise ValueError(
                f"{expected_schema} artifact bindings changed: {mismatches}."
            )
    return bindings


def validate_learning_probe_contract(
    preflight_decision: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    bindings = validate_stage_decision(
        preflight_decision,
        expected_schema=PREFLIGHT_DECISION_SCHEMA,
        expected_bindings=expected_bindings,
    )
    schedule = preflight_decision.get("allowed_schedule")
    if not isinstance(schedule, Sequence) or len(schedule) != 5:
        raise ValueError("Preflight decision has no five-point probe schedule.")
    points = []
    for point in schedule:
        if not isinstance(point, Sequence) or len(point) != 2:
            raise ValueError("Every probe schedule point must be [fraction, weight].")
        fraction, weight = float(point[0]), float(point[1])
        if not 0.0 <= fraction <= 1.0 or not 0.0 <= weight <= 0.5:
            raise ValueError("Probe schedule fraction/weight is outside its contract.")
        points.append((fraction, weight))
    if tuple(fraction for fraction, _ in points) != (0.0, 0.1, 0.3, 0.6, 1.0):
        raise ValueError("Probe schedule fractions do not match the preregistration.")
    w0 = float(preflight_decision.get("w0"))
    w_cap = float(preflight_decision.get("w_cap"))
    if not 0.0 < w0 <= 0.05 or not 0.2 <= w_cap <= 0.5:
        raise ValueError("Preflight-selected w0/w_cap is outside the allowed range.")
    expected_schedule = (
        (0.0, w0),
        (0.1, w0),
        (0.3, min(0.2, w_cap)),
        (0.6, w_cap),
        (1.0, w_cap),
    )
    if tuple(points) != expected_schedule:
        raise ValueError("Probe schedule does not match w0/w_cap.")
    return {
        "artifact_bindings": bindings,
        "w0": w0,
        "w_cap": w_cap,
        "schedule": [list(point) for point in points],
    }


def validate_formal_training_contract(
    *,
    preflight_decision: Mapping[str, Any],
    learning_probe_decision: Mapping[str, Any],
    lineage_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = validate_learning_probe_contract(preflight_decision)
    probe_bindings = validate_stage_decision(
        learning_probe_decision,
        expected_schema=LEARNING_PROBE_DECISION_SCHEMA,
        expected_bindings=preflight["artifact_bindings"],
    )
    lineage_artifacts = validate_e_i_lineage_manifest(lineage_manifest)
    lineage_sha = {
        name: record["sha256"] for name, record in lineage_artifacts.items()
    }
    mismatches = {
        name: (probe_bindings.get(name), sha)
        for name, sha in lineage_sha.items()
        if probe_bindings.get(name) != sha
    }
    if mismatches:
        raise ValueError(
            "Preflight/probe decisions do not bind the validated E-I lineage: "
            f"{mismatches}."
        )
    if learning_probe_decision.get("initializer_stage") != "e_i_s0":
        raise ValueError("Formal training cannot continue from Canary/probe weights.")
    return {
        "artifact_bindings": probe_bindings,
        "schedule": preflight["schedule"],
        "base_learning_rate": 1e-5,
        "num_epochs": 10,
        "initializer_stage": "e_i_s0",
    }
