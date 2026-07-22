#!/usr/bin/env python3
"""Audit E-I lineage and enforce S-DR stage transition contracts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastwam.adaptive_gate.sdr_contracts import (
    LEARNING_PROBE_DECISION_SCHEMA,
    FORMAL_TRAINING_DECISION_SCHEMA,
    PREFLIGHT_DECISION_SCHEMA,
    artifact_record,
    atomic_json,
    audit_e_i_lineage_inputs,
    read_json,
    validate_e_i_lineage_manifest,
    validate_formal_training_contract,
    validate_learning_probe_contract,
    validate_stage_decision,
)


def audit_lineage(args: argparse.Namespace) -> None:
    result = audit_e_i_lineage_inputs(
        wan_robot_base_checkpoint=args.wan_robot_base_checkpoint,
        wan_robot_base_config=args.wan_robot_base_config,
        e_i_checkpoint=args.e_i_checkpoint,
        e_i_config=args.e_i_config,
        dataset_stats=args.dataset_stats,
        lineage_manifest=args.lineage_manifest,
    )
    atomic_json(args.out, result)
    print(json.dumps(result, sort_keys=True))


def check_lineage(args: argparse.Namespace) -> None:
    manifest = read_json(args.lineage_manifest)
    expected = {
        "wan_robot_base_checkpoint": args.wan_robot_base_checkpoint,
        "wan_robot_base_config": args.wan_robot_base_config,
        "e_i_checkpoint": args.e_i_checkpoint,
        "e_i_config": args.e_i_config,
        "dataset_stats": args.dataset_stats,
    }
    artifacts = validate_e_i_lineage_manifest(manifest, expected_paths=expected)
    print(
        json.dumps(
            {
                "status": "PASS",
                "artifact_sha256": {
                    name: value["sha256"] for name, value in artifacts.items()
                },
            },
            sort_keys=True,
        )
    )


def check_probe(args: argparse.Namespace) -> None:
    result = validate_learning_probe_contract(read_json(args.preflight_decision))
    print(json.dumps({"status": "PASS", **result}, sort_keys=True))


def check_formal(args: argparse.Namespace) -> None:
    result = validate_formal_training_contract(
        preflight_decision=read_json(args.preflight_decision),
        learning_probe_decision=read_json(args.learning_probe_decision),
        lineage_manifest=read_json(args.lineage_manifest),
    )
    print(json.dumps({"status": "PASS", **result}, sort_keys=True))


def check_decision(args: argparse.Namespace) -> None:
    validate_stage_decision(
        read_json(args.decision),
        expected_schema=args.schema,
    )
    print(json.dumps({"status": "PASS", "schema": args.schema}, sort_keys=True))


def print_schedule(args: argparse.Namespace) -> None:
    result = validate_learning_probe_contract(read_json(args.preflight_decision))
    schedule = [
        {"fraction": fraction, "weight": weight}
        for fraction, weight in result["schedule"]
    ]
    print(json.dumps(schedule, separators=(",", ":"), allow_nan=False))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_formal_manifest(args: argparse.Namespace) -> None:
    preflight_path = Path(args.preflight_decision).expanduser().resolve()
    probe_path = Path(args.learning_probe_decision).expanduser().resolve()
    lineage_path = Path(args.lineage_manifest).expanduser().resolve()
    preflight = read_json(preflight_path)
    contract = validate_formal_training_contract(
        preflight_decision=preflight,
        learning_probe_decision=read_json(probe_path),
        lineage_manifest=read_json(lineage_path),
    )
    repo = Path(args.repo).expanduser().resolve()
    outer_repo = Path(args.outer_repo).expanduser().resolve()
    commit = _git_value(repo, "rev-parse", "HEAD")
    dirty = bool(_git_value(repo, "status", "--porcelain"))
    code_sha = hashlib.sha256(str(commit).encode("ascii")).hexdigest()
    expected_code_sha = contract["artifact_bindings"].get("code_commit")
    if dirty or code_sha != expected_code_sha:
        raise ValueError(
            "Formal run code differs from the clean preflight commit: "
            f"dirty={dirty}, current={commit}, expected_sha={expected_code_sha}."
        )

    verified_bindings = {}
    raw_bindings = preflight.get("artifact_bindings")
    for name, record in raw_bindings.items():
        if name == "code_commit":
            verified_bindings[name] = {
                "path": f"git:{commit}",
                "sha256": code_sha,
            }
            continue
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        actual = artifact_record(path)
        if actual["sha256"] != record.get("sha256"):
            raise ValueError(f"Formal input artifact changed: {name}={path}.")
        verified_bindings[str(name)] = actual

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_dir)
    safety_bytes = int(args.disk_safety_bytes)
    if disk.free < safety_bytes:
        raise ValueError(
            "Formal run disk preflight failed: "
            f"free={disk.free}, required={safety_bytes}."
        )
    try:
        import torch

        torch_environment = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "gpu_total_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else None
            ),
        }
    except Exception as exc:
        torch_environment = {"inspection_error": str(exc)}

    result = {
        "schema": "fastwam-sdr-formal-run-manifest-v1",
        "status": "AUTHORIZED",
        "plus_full_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_bindings": verified_bindings,
        "stage_inputs": {
            "preflight_decision": artifact_record(preflight_path),
            "learning_probe_decision": artifact_record(probe_path),
            "lineage_manifest": artifact_record(lineage_path),
            "resolved_config": artifact_record(args.resolved_config),
            "step_contract": artifact_record(args.step_contract),
            "validation_manifest": artifact_record(args.validation_manifest),
            "warmstart_decision": artifact_record(args.warmstart_decision),
            "launcher": artifact_record(args.launcher),
        },
        "formal_contract": contract,
        "repositories": {
            "fastwam": {
                "path": str(repo),
                "commit": commit,
                "branch": _git_value(repo, "branch", "--show-current"),
                "dirty": dirty,
            },
            "outer": {
                "path": str(outer_repo),
                "commit": _git_value(outer_repo, "rev-parse", "HEAD"),
                "branch": _git_value(outer_repo, "branch", "--show-current"),
                "dirty": bool(
                    _git_value(outer_repo, "status", "--porcelain")
                ),
                "submodule_status": _git_value(
                    outer_repo, "submodule", "status", "FastWAM"
                ),
            },
        },
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "platform": platform.platform(),
            "accelerate": _package_version("accelerate"),
            "deepspeed": _package_version("deepspeed"),
            **torch_environment,
        },
        "disk_preflight": {
            "path": str(output_dir),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "safety_floor_bytes": safety_bytes,
            "pass": True,
        },
        "no_plus_full_statement": (
            "No Plus-Full or task rollout outcome was loaded or used."
        ),
    }
    atomic_json(args.out, result)
    print(json.dumps({"status": "AUTHORIZED", "git_commit": commit}, sort_keys=True))


def _git_value(cwd: Path, *command: str) -> str | None:
    result = subprocess.run(
        ["git", *command],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def record_not_run(args: argparse.Namespace) -> None:
    import yaml

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lineage_audit = read_json(args.lineage_audit)
    if lineage_audit.get("status") == "PASS":
        raise ValueError("record-not-run refuses a PASS lineage audit.")
    blockers = list(lineage_audit.get("blockers") or ())
    if not blockers:
        blockers = ["E-I lineage audit did not pass."]
    validation_manifest = Path(args.validation_manifest).expanduser().resolve()
    bindings = dict(lineage_audit.get("artifacts") or {})
    bindings["validation_manifest"] = artifact_record(validation_manifest)
    bindings["lineage_audit"] = artifact_record(args.lineage_audit)
    if args.warmstart_decision:
        bindings["warmstart_decision"] = artifact_record(args.warmstart_decision)

    now = datetime.now(timezone.utc).isoformat()
    common = {
        "status": "NOT-RUN",
        "generated_at_utc": now,
        "blockers": blockers,
        "plus_full_used": False,
    }
    gradient = {
        "schema": "fastwam-sdr-gradient-diagnostics-v1",
        **common,
        "optimizer_steps": 0,
        "common_replay_count": 0,
        "independent_replay_count": 0,
        "exact_effective_batch_gradient": None,
        "reason": "Stopped before model load because the E-I lineage gate failed.",
    }
    generated = {
        "schema": "fastwam-sdr-generated-future-validation-v1",
        **common,
        "cache_created": False,
        "sample_count": 0,
        "action_sensitivity_claim": (
            "No result. Action sensitivity would be conditioning-read evidence, "
            "not task usefulness evidence."
        ),
    }
    preflight = {
        "schema": PREFLIGHT_DECISION_SCHEMA,
        **common,
        "artifact_bindings": bindings,
        "w0": None,
        "w_cap": None,
        "safe_candidate_weights": [],
        "allowed_schedule": None,
        "next_stage_authorized": False,
    }
    probe = {
        "schema": LEARNING_PROBE_DECISION_SCHEMA,
        **common,
        "artifact_bindings": bindings,
        "initializer_stage": "e_i_s0",
        "canary": "NOT-RUN",
        "probe_500_step": "NOT-RUN",
        "next_stage_authorized": False,
    }
    formal = {
        "schema": FORMAL_TRAINING_DECISION_SCHEMA,
        **common,
        "artifact_bindings": bindings,
        "initializer_stage": "e_i_s0",
        "formal_training": "NOT-RUN",
        "final_checkpoint": None,
        "s_dr_selection": None,
    }
    atomic_json(output_dir / "gradient_diagnostics.json", gradient)
    atomic_json(output_dir / "generated_future_validation.json", generated)
    atomic_json(output_dir / "preflight_decision.json", preflight)
    atomic_json(output_dir / "learning_probe_decision.json", probe)
    atomic_json(output_dir / "formal_training_decision.json", formal)

    reconstructed = (
        artifact_record(args.reconstructed_e_i_config)
        if args.reconstructed_e_i_config
        else None
    )
    resolved = {
        "schema": "fastwam-sdr-resolved-config-v1",
        "status": "NOT-RUN",
        "model": {
            "kind": "FusedDualRegimeFastWAM",
            "shared_action_dit": True,
            "mode_specific_parameters": False,
        },
        "training": {
            "base_learning_rate": 1e-5,
            "microbatch_size": 1,
            "gradient_accumulation_steps": 64,
            "global_batch_size": 64,
            "mixed_precision": "bf16",
            "video_lr_scale": 0.0,
            "proprio_lr_scale": 0.0,
            "action_lr_scale": 1.0,
            "main_action_noise_coupling": "independent",
            "diagnostic_action_noise_coupling": [
                "common",
                "independent",
            ],
        },
        "validation": {
            "manifest": str(validation_manifest),
            "seed": 20260721,
            "video_solver_steps": 20,
            "action_solver_steps": 20,
        },
        "lineage_audit": str(Path(args.lineage_audit).resolve()),
        "nonadmissible_reconstructed_e_i_config": reconstructed,
        "blockers": blockers,
    }
    resolved_path = output_dir / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved, sort_keys=True),
        encoding="utf-8",
    )

    repo = Path(args.repo).expanduser().resolve()
    usage = shutil.disk_usage(output_dir)
    gpu = {}
    try:
        import torch

        gpu = {
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except Exception as exc:
        gpu = {"inspection_error": str(exc)}
    run_manifest = {
        "schema": "fastwam-sdr-preflight-run-manifest-v1",
        **common,
        "repo": str(repo),
        "git_commit": _git_value(repo, "rev-parse", "HEAD"),
        "git_branch": _git_value(repo, "branch", "--show-current"),
        "git_dirty": bool(_git_value(repo, "status", "--porcelain")),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "gpu": gpu,
        "disk": {
            "path": str(output_dir),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
        "artifacts": {
            "lineage_audit": artifact_record(args.lineage_audit),
            "validation_manifest": artifact_record(validation_manifest),
            "resolved_config": artifact_record(resolved_path),
            "gradient_diagnostics": artifact_record(
                output_dir / "gradient_diagnostics.json"
            ),
            "generated_future_validation": artifact_record(
                output_dir / "generated_future_validation.json"
            ),
            "preflight_decision": artifact_record(
                output_dir / "preflight_decision.json"
            ),
            "learning_probe_decision": artifact_record(
                output_dir / "learning_probe_decision.json"
            ),
            "formal_training_decision": artifact_record(
                output_dir / "formal_training_decision.json"
            ),
        },
        "stages": {
            "e1_p0_5": "NOT-RUN",
            "canary_50": "NOT-RUN",
            "probe_500": "NOT-RUN",
            "e1_p2_formal": "NOT-RUN",
        },
    }
    atomic_json(output_dir / "run_manifest.json", run_manifest)
    print(
        json.dumps(
            {
                "status": "NOT-RUN",
                "output_dir": str(output_dir),
                "blockers": blockers,
            },
            sort_keys=True,
        )
    )


def record_formal_stop(args: argparse.Namespace) -> None:
    output = Path(args.out).expanduser().resolve()
    if output.exists():
        print(json.dumps({"status": "UNCHANGED", "path": str(output)}))
        return
    preflight = read_json(args.preflight_decision)
    status = str(args.status)
    result = {
        "schema": FORMAL_TRAINING_DECISION_SCHEMA,
        "status": status,
        "plus_full_used": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_bindings": preflight.get("artifact_bindings", {}),
        "initializer_stage": "e_i_s0",
        "formal_training": status,
        "failure_conditions": [str(args.reason)],
        "final_checkpoint": None,
        "s_dr_selection": None,
        "stage_inputs": {
            "preflight_decision": artifact_record(args.preflight_decision),
            "learning_probe_decision": artifact_record(
                args.learning_probe_decision
            ),
            "lineage_manifest": artifact_record(args.lineage_manifest),
        },
        "no_plus_full_statement": (
            "No Plus-Full or task rollout outcome was loaded or used."
        ),
    }
    atomic_json(output, result)
    print(json.dumps({"status": status, "path": str(output)}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-lineage")
    audit.add_argument("--wan-robot-base-checkpoint")
    audit.add_argument("--wan-robot-base-config")
    audit.add_argument("--e-i-checkpoint")
    audit.add_argument("--e-i-config")
    audit.add_argument("--dataset-stats")
    audit.add_argument("--lineage-manifest")
    audit.add_argument("--out", required=True)
    audit.set_defaults(func=audit_lineage)
    lineage = sub.add_parser("check-lineage")
    lineage.add_argument("--wan-robot-base-checkpoint", required=True)
    lineage.add_argument("--wan-robot-base-config", required=True)
    lineage.add_argument("--e-i-checkpoint", required=True)
    lineage.add_argument("--e-i-config", required=True)
    lineage.add_argument("--dataset-stats", required=True)
    lineage.add_argument("--lineage-manifest", required=True)
    lineage.set_defaults(func=check_lineage)
    probe = sub.add_parser("check-probe")
    probe.add_argument("--preflight-decision", required=True)
    probe.set_defaults(func=check_probe)
    formal = sub.add_parser("check-formal")
    formal.add_argument("--preflight-decision", required=True)
    formal.add_argument("--learning-probe-decision", required=True)
    formal.add_argument("--lineage-manifest", required=True)
    formal.set_defaults(func=check_formal)
    decision = sub.add_parser("check-decision")
    decision.add_argument(
        "--schema",
        choices=[PREFLIGHT_DECISION_SCHEMA, LEARNING_PROBE_DECISION_SCHEMA],
        required=True,
    )
    decision.add_argument("--decision", required=True)
    decision.set_defaults(func=check_decision)
    schedule = sub.add_parser("print-schedule")
    schedule.add_argument("--preflight-decision", required=True)
    schedule.set_defaults(func=print_schedule)
    formal_manifest = sub.add_parser("write-formal-manifest")
    formal_manifest.add_argument("--preflight-decision", required=True)
    formal_manifest.add_argument("--learning-probe-decision", required=True)
    formal_manifest.add_argument("--lineage-manifest", required=True)
    formal_manifest.add_argument("--resolved-config", required=True)
    formal_manifest.add_argument("--step-contract", required=True)
    formal_manifest.add_argument("--validation-manifest", required=True)
    formal_manifest.add_argument("--warmstart-decision", required=True)
    formal_manifest.add_argument("--launcher", required=True)
    formal_manifest.add_argument("--repo", required=True)
    formal_manifest.add_argument("--outer-repo", required=True)
    formal_manifest.add_argument("--output-dir", required=True)
    formal_manifest.add_argument("--disk-safety-bytes", type=int, required=True)
    formal_manifest.add_argument("--out", required=True)
    formal_manifest.set_defaults(func=write_formal_manifest)
    not_run = sub.add_parser("record-not-run")
    not_run.add_argument("--lineage-audit", required=True)
    not_run.add_argument("--validation-manifest", required=True)
    not_run.add_argument("--warmstart-decision")
    not_run.add_argument("--reconstructed-e-i-config")
    not_run.add_argument("--repo", required=True)
    not_run.add_argument("--output-dir", required=True)
    not_run.set_defaults(func=record_not_run)
    formal_stop = sub.add_parser("record-formal-stop")
    formal_stop.add_argument(
        "--status",
        choices=["NOT-RUN", "FAIL-DIAGNOSED"],
        required=True,
    )
    formal_stop.add_argument("--reason", required=True)
    formal_stop.add_argument("--preflight-decision", required=True)
    formal_stop.add_argument("--learning-probe-decision", required=True)
    formal_stop.add_argument("--lineage-manifest", required=True)
    formal_stop.add_argument("--out", required=True)
    formal_stop.set_defaults(func=record_formal_stop)
    return ap


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
