"""Contract tests for the E1-P1D-LC low-cap (``w_cap=0.09``) Canary tooling.

Every fixture is synthetic. Nothing here loads a real checkpoint, GPU artifact,
simulator rollout or Plus-Full outcome. The one real file that is reused is the
archived ``FAIL-DIAGNOSED`` E1-P1 ``canary_decision.json``: it is read and hashed
so the pinned trigger SHA256 is exercised against the actual archived bytes.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from fastwam.adaptive_gate.sdr_contracts import (
    ARCHIVED_FAILED_CANARY_DECISION_SHA256,
    ARCHIVED_P0_5_W0,
    CANARY_DECISION_SCHEMA,
    LOW_CAP_CANARY_DECISION_SCHEMA,
    LOW_CAP_INVARIANT_BINDING_NAMES,
    LOW_CAP_NEW_W_CAP,
    LOW_CAP_PREREGISTRATION_SCHEMA,
    PREFLIGHT_DECISION_SCHEMA,
    float32,
    low_cap_locked_schedule,
    validate_failed_canary_trigger,
    validate_learning_probe_contract,
    validate_low_cap_invariant_bindings,
    validate_low_cap_uncond_weight_schedule,
)
from fastwam.adaptive_gate.sdr_preflight import weighted_descent_margins
from fastwam.adaptive_gate.training import uncond_weight_at_step


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOW_CAP_LAUNCHER = PROJECT_ROOT / "scripts/adaptive_gate/run_e1_sdr_low_cap_canary.sh"
PROBE_LAUNCHER = PROJECT_ROOT / "scripts/adaptive_gate/run_e1_sdr_learning_probe.sh"
ARCHIVED_CANARY = (
    PROJECT_ROOT
    / "docs/validation/e1/20260723T034745Z_sdr_canary_no_go/canary_decision.json"
)
GRID_WEIGHTS = (0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0)
W0 = ARCHIVED_P0_5_W0
CONDITION_COUNT = 18

# Real archived step-50 common-noise action-all statistics. The persisted weight
# grid has no 0.09 entry, so the decision must recompute the margins from these.
REAL_STEP50_COMMON_ACTION_ALL = (
    0.052151287778994644,
    5.526134393182538,
    -0.37523997115484053,
)
INDEPENDENT_ACTION_ALL = (0.03, 4.0, -0.2)
FINAL_BLOCK_AGGREGATE = (0.01, 1.0, -0.05)
# One of eight shards has a negative IDM margin at w=0.09 -> fraction 0.125.
COMMON_ACTION_ALL_SHARDS = [(0.05, 5.5, -0.3)] * 7 + [(0.01, 5.5, -0.3)]
FINAL_BLOCK_SHARDS = [(0.01, 1.0, -0.05)] * 8

STEP_PROFILE = {
    0: dict(
        uncond=0.2864014894,
        idm=0.0046307122,
        gt=0.0238642551,
        valid=0.1068509231,
        sensitivity=0.4067696333,
    ),
    10: dict(uncond=0.2500, idm=0.00465, gt=0.0240, valid=0.1075, sensitivity=0.4060),
    25: dict(uncond=0.2100, idm=0.00468, gt=0.0242, valid=0.1080, sensitivity=0.4050),
    50: dict(
        uncond=0.1739948267,
        idm=0.00470,
        gt=0.0245,
        valid=0.1090,
        sensitivity=0.4042735845,
    ),
}


def _decision_module():
    script = PROJECT_ROOT / "scripts/decide_sdr_low_cap_canary.py"
    spec = importlib.util.spec_from_file_location("sdr_low_cap_decision", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _decision_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _group(idm_sq: float, uncond_sq: float, dot: float) -> dict:
    stats = {
        "idm_sq": idm_sq,
        "uncond_sq": uncond_sq,
        "dot": dot,
        "idm_norm": math.sqrt(idm_sq),
        "uncond_norm": math.sqrt(uncond_sq),
        "cosine": dot / math.sqrt(idm_sq * uncond_sq),
        "parameter_tensor_count": 4,
    }
    stats["margins_by_weight"] = {
        str(weight): weighted_descent_margins(stats, weight) for weight in GRID_WEIGHTS
    }
    return stats


def _side(coupling: str, action_all: tuple[float, float, float]) -> dict:
    return {
        "coupling": coupling,
        "sample_count": 40,
        "groups": {
            "action_all": _group(*action_all),
            "action_blocks_final": _group(*FINAL_BLOCK_AGGREGATE),
        },
        "shards": [
            {
                "shard_index": index,
                "sample_count": 5,
                "groups": {
                    "action_all": _group(*action_shard),
                    "action_blocks_final": _group(*final_shard),
                },
            }
            for index, (action_shard, final_shard) in enumerate(
                zip(COMMON_ACTION_ALL_SHARDS, FINAL_BLOCK_SHARDS)
            )
        ],
    }


def _write_diagnostics(directory: Path, step: int, **overrides) -> None:
    profile = {**STEP_PROFILE[step], **overrides}
    directory.mkdir(parents=True, exist_ok=True)
    gradient = {
        "schema": "fastwam-sdr-gradient-diagnostics-v1",
        "status": "PASS",
        "optimizer_steps": step,
        "loss_records": [
            {
                "coupling": coupling,
                "shard_index": index,
                "raw_idm": profile["idm"],
                "raw_uncond": profile["uncond"],
            }
            for coupling in ("common", "independent")
            for index in range(8)
        ],
        "common": _side("common", REAL_STEP50_COMMON_ACTION_ALL),
        "independent": _side("independent", INDEPENDENT_ACTION_ALL),
    }
    (directory / "gradient_diagnostics.json").write_text(
        json.dumps(gradient), encoding="utf-8"
    )
    conditions = {
        "gt_teacher_forced_future": profile["gt"],
        "valid_self_generated_future": profile["valid"],
        "no_read": 0.31,
        "repeat_current": 0.28,
        "matched_shuffled_future": 0.33,
        "forced_uncond": 0.30,
        "standalone_e_i_reference": 0.02,
    }
    generated = {
        "schema": "fastwam-sdr-generated-future-validation-v1",
        "status": "PASS",
        "cache_created": True,
        "direct_generated_cached_valid_parity": {"pass": True},
        "sample_count": 4,
        "records": [
            {
                "sample_id": f"sample-{index}",
                "conditions": {
                    name: {"normalized_error": {"l1": value / 2, "l2": value}}
                    for name, value in conditions.items()
                },
            }
            for index in range(4)
        ],
        "sensitivity_gate": {
            "median": profile["sensitivity"],
            "pass": True,
            "threshold": 0.001,
        },
        "no_read_uncond_parity": {"max_abs": 0.0, "tolerance": 1e-4, "pass": True},
    }
    (directory / "generated_future_validation.json").write_text(
        json.dumps(generated), encoding="utf-8"
    )


def _ledger_rows(
    *,
    w0: float = W0,
    steps: int = 50,
    cap: float = LOW_CAP_NEW_W_CAP,
    clipped_steps: int = 0,
) -> list[dict]:
    schedule = [[0.0, w0], [0.1, w0], [0.3, cap], [0.6, cap], [1.0, cap]]
    rows = []
    for index in range(steps):
        weight = float32(
            uncond_weight_at_step(
                schedule, optimizer_step=index, total_optimizer_steps=steps
            )
        )
        rows.append(
            {
                "schema": "fastwam-training-metric-v1",
                "global_step": index + 1,
                "dual_regime_optimizer_steps": index + 1,
                "epoch": 0,
                "gradient_clipped": index < clipped_steps,
                "optimizer_step_was_skipped": False,
                "grad_norm_before_clip": 0.11,
                "max_grad_norm": 1.0,
                "learning_rate": 1e-5 * (1.0 - index / 100.0),
                "loss": 0.007,
                "losses": {
                    "action_regime_weight_uncond": weight,
                    "dual_regime_optimizer_steps": float(index + 1),
                    "dual_regime_schedule_fraction": (index + 1) / steps,
                    "loss_action_combined": 0.0028,
                    "loss_action_idm_raw": 0.0036,
                    "loss_action_uncond_raw": 0.29,
                    "loss_video": 0.004,
                },
                "peak_gpu_memory_bytes": 33_000_000_000,
                "elapsed_seconds": 30.0 * (index + 1),
                "step_duration_seconds": 30.0,
                "samples_per_second": 2.1,
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _git_repo(root: Path) -> tuple[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "file.txt").write_text("code", encoding="utf-8")
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, env=env)
    subprocess.run(["git", "add", "."], cwd=root, check=True, env=env)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--no-gpg-sign",
            "-m",
            "c",
        ],
        cwd=root,
        check=True,
        env=env,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()
    return commit, hashlib.sha256(commit.encode("ascii")).hexdigest()


class Case:
    """A complete synthetic low-cap run laid out on disk."""

    def __init__(self, root: Path):
        self.root = root
        self.repo = root / "repo"
        self.commit, self.code_sha = _git_repo(self.repo)

        self.artifacts = root / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, Path] = {}
        for name in LOW_CAP_INVARIANT_BINDING_NAMES:
            path = self.artifacts / f"{name}.bin"
            path.write_text(f"{name}-payload", encoding="utf-8")
            self.files[name] = path

        self.baseline = root / "fresh_preflight"
        self.baseline.mkdir(parents=True, exist_ok=True)
        self.preflight_resolved = self.baseline / "resolved_config.yaml"
        self.preflight_resolved.write_text("placeholder: true\n", encoding="utf-8")

        self.bindings = {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in self.files.items()
        }
        self.bindings["resolved_config"] = {
            "path": str(self.preflight_resolved),
            "sha256": _sha256(self.preflight_resolved),
        }
        self.bindings["code_commit"] = {
            "path": f"git:{self.commit}",
            "sha256": self.code_sha,
        }

        self.preflight_path = self.baseline / "preflight_decision.json"
        self.failed_path = root / "failed_canary" / "canary_decision.json"
        self.failed_path.parent.mkdir(parents=True, exist_ok=True)
        self.reference_ledger = (
            self.failed_path.parent / "canary_training_metrics.jsonl"
        )

        self.run_dir = root / "run"
        self.train_dir = self.run_dir / "train"
        self.decision_path = self.run_dir / "low_cap_canary_decision.json"
        self.preregistration_path = self.run_dir / "low_cap_preregistration.json"
        self.resolved_config_path = self.run_dir / "low_cap_resolved_config.yaml"
        self.training_metrics = self.train_dir / "training_metrics.jsonl"
        self.plan = root / "post_canary_low_cap_execution_plan.md"
        self.plan.write_text("# plan\n", encoding="utf-8")

        self.w0 = W0
        self.cap = LOW_CAP_NEW_W_CAP
        self.diagnostics: dict[int, Path] = {}
        self.deltas: dict[int, Path] = {}

    # -- writers -----------------------------------------------------------
    def write_preflight(self, **overrides) -> None:
        payload = {
            "schema": PREFLIGHT_DECISION_SCHEMA,
            "status": "PASS",
            "plus_full_used": False,
            "artifact_bindings": copy.deepcopy(self.bindings),
            "w0": self.w0,
            "w_cap": 0.5,
            "allowed_schedule": [
                [0.0, self.w0],
                [0.1, self.w0],
                [0.3, 0.2],
                [0.6, 0.5],
                [1.0, 0.5],
            ],
            "next_stage_authorized": True,
        }
        payload.update(overrides)
        self.preflight_path.write_text(json.dumps(payload), encoding="utf-8")

    def write_failed_canary(self, **overrides) -> None:
        _write_jsonl(self.reference_ledger, _ledger_rows())
        payload = {
            "schema": CANARY_DECISION_SCHEMA,
            "status": "FAIL-DIAGNOSED",
            "plus_full_used": False,
            "initializer_stage": "e_i_s0",
            "probe_500_authorized": False,
            "failure_conditions": ["common-noise IDM margin is not positive"],
            "artifact_bindings": copy.deepcopy(self.bindings),
            "artifacts": {
                "training_metrics": {
                    "path": str(self.reference_ledger),
                    "sha256": _sha256(self.reference_ledger),
                    "size_bytes": self.reference_ledger.stat().st_size,
                }
            },
        }
        payload.update(overrides)
        self.failed_path.write_text(json.dumps(payload), encoding="utf-8")

    def write_run(
        self,
        *,
        config_overrides: dict | None = None,
        ledger_rows: list[dict] | None = None,
        diagnostic_overrides: dict[int, dict] | None = None,
    ) -> None:
        _write_jsonl(self.training_metrics, ledger_rows or _ledger_rows())
        diagnostic_overrides = diagnostic_overrides or {}
        for step in (0, 10, 25, 50):
            directory = (
                self.baseline if step == 0 else self.run_dir / f"diagnostics_step{step}"
            )
            _write_diagnostics(directory, step, **diagnostic_overrides.get(step, {}))
            self.diagnostics[step] = directory
        weights = self.train_dir / "checkpoints/weights"
        weights.mkdir(parents=True, exist_ok=True)
        for step in (10, 25, 50):
            delta = weights / f"step_{step:06d}.action_dit_delta.pt"
            delta.write_bytes(f"delta-{step}".encode("ascii"))
            self.deltas[step] = delta
        config = {
            "learning_rate": 1e-5,
            "max_steps": 50,
            "seed": 42,
            "batch_size": 1,
            "gradient_accumulation_steps": 64,
            "mixed_precision": "bf16",
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "weights_checkpoint_kind": "action_dit_delta",
            "dual_regime_training": {
                "uncond_weight_schedule": [
                    [0.0, self.w0],
                    [0.1, self.w0],
                    [0.3, self.cap],
                    [0.6, self.cap],
                    [1.0, self.cap],
                ],
                "optimizer": {
                    "action_lr_scale": 1.0,
                    "proprio_lr_scale": 0.0,
                    "video_lr_scale": 0.0,
                },
            },
            "data": {
                "train": {
                    "pretrained_norm_stats": str(self.files["dataset_stats"]),
                    "episode_split_manifest": str(self.files["validation_manifest"]),
                    "manifest_split": "train",
                }
            },
        }
        for dotted, value in (config_overrides or {}).items():
            target = config
            parts = dotted.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        self.resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
        self.resolved_config_path.write_text(
            yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
        )

    # -- drivers -----------------------------------------------------------
    def preregister(self) -> dict:
        args = MODULE.parser().parse_args(
            [
                "preregister",
                "--preflight-decision",
                str(self.preflight_path),
                "--failed-canary-decision",
                str(self.failed_path),
                "--low-cap-plan",
                str(self.plan),
                "--expected-failed-canary-sha256",
                _sha256(self.failed_path),
                "--out",
                str(self.preregistration_path),
            ]
        )
        MODULE.preregister(args)
        return json.loads(self.preregistration_path.read_text(encoding="utf-8"))

    def decide_args(self, *extra: str):
        return MODULE.parser().parse_args(
            [
                "decide",
                "--preflight-decision",
                str(self.preflight_path),
                "--failed-canary-decision",
                str(self.failed_path),
                "--low-cap-plan",
                str(self.plan),
                "--expected-failed-canary-sha256",
                _sha256(self.failed_path),
                "--preregistration",
                str(self.preregistration_path),
                "--resolved-config",
                str(self.resolved_config_path),
                "--baseline-diagnostics",
                str(self.diagnostics[0]),
                "--step10-diagnostics",
                str(self.diagnostics[10]),
                "--step25-diagnostics",
                str(self.diagnostics[25]),
                "--step50-diagnostics",
                str(self.diagnostics[50]),
                "--training-metrics",
                str(self.training_metrics),
                "--step10-delta",
                str(self.deltas[10]),
                "--step25-delta",
                str(self.deltas[25]),
                "--step50-delta",
                str(self.deltas[50]),
                "--repo",
                str(self.repo),
                "--out",
                str(self.decision_path),
                *extra,
            ]
        )

    def build(self, *extra: str) -> dict:
        return MODULE.build_low_cap_decision(self.decide_args(*extra))

    def run_cli(self, *extra: str) -> dict:
        with pytest.raises(SystemExit) as exit_info:
            MODULE.decide(self.decide_args(*extra))
        assert exit_info.value.code == 3
        return json.loads(self.decision_path.read_text(encoding="utf-8"))


@pytest.fixture()
def case(tmp_path) -> Case:
    instance = Case(tmp_path)
    instance.write_preflight()
    instance.write_failed_canary()
    instance.write_run()
    instance.preregister()
    return instance


# ---------------------------------------------------------------------------
# Happy path and the 18 preregistered acceptance conditions
# ---------------------------------------------------------------------------
def test_complete_synthetic_low_cap_run_passes(case):
    decision = case.build()

    assert decision["schema"] == LOW_CAP_CANARY_DECISION_SCHEMA
    assert decision["status"] == "PASS"
    assert decision["stage"] == "E1-P1D-LC"
    assert decision["failure_conditions"] == []
    assert decision["contract_violations"] == []
    assert decision["probe_500_authorized"] is True
    assert decision["formal_training_authorized"] is False
    assert decision["initializer_stage"] == "e_i_s0"
    assert decision["single_variable"]["old_w_cap"] == 0.5
    assert decision["single_variable"]["new_w_cap"] == 0.09


def test_decision_reports_all_eighteen_named_conditions(case):
    conditions = case.build()["acceptance_conditions"]

    assert len(conditions) == CONDITION_COUNT
    assert [name[:12] for name in sorted(conditions)] == [
        f"condition_{index:02d}" for index in range(1, CONDITION_COUNT + 1)
    ]
    assert all(conditions.values())


def test_margins_are_recomputed_at_0_09_from_persisted_statistics(case):
    gradient = json.loads(
        (case.diagnostics[50] / "gradient_diagnostics.json").read_text(encoding="utf-8")
    )
    assert "0.09" not in gradient["common"]["groups"]["action_all"]["margins_by_weight"]

    common = case.build()["evaluation_steps"]["50"]["common"]

    assert common["weight"] == 0.09
    assert common["action_all"]["idm_margin"] == pytest.approx(0.016862101, abs=1e-9)
    assert common["action_all"]["uncond_margin"] == pytest.approx(0.112029472, abs=1e-9)
    assert common["action_all"]["weighted_gradient_norm_ratio"] == pytest.approx(
        0.926447283, abs=1e-9
    )
    assert common["action_all"]["negative_idm_shard_fraction"] == 0.125
    assert common["action_all"]["negative_uncond_shard_fraction"] == 0.0
    assert common["action_blocks_final"]["negative_idm_shard_fraction"] == 0.0
    assert common["recomputation_cross_check"]["worst_absolute_difference"] == 0.0


def test_condition_01_requires_exactly_fifty_applied_steps(case):
    case.write_run(ledger_rows=_ledger_rows(steps=49))

    decision = case.build()

    assert decision["status"] == "FAIL-DIAGNOSED"
    assert (
        decision["acceptance_conditions"][
            "condition_01_fifty_applied_steps_no_skip_no_nonfinite"
        ]
        is False
    )


def test_condition_01_rejects_a_skipped_optimizer_step(case):
    rows = _ledger_rows()
    rows[20]["optimizer_step_was_skipped"] = True
    case.write_run(ledger_rows=rows)

    decision = case.build()

    assert (
        decision["acceptance_conditions"][
            "condition_01_fifty_applied_steps_no_skip_no_nonfinite"
        ]
        is False
    )
    assert decision["training"]["skipped_optimizer_steps"] == 1


def test_condition_01_rejects_a_non_finite_ledger_value(case):
    rows = _ledger_rows()
    rows[5]["grad_norm_before_clip"] = float("nan")
    path = case.training_metrics
    path.write_text(
        "".join(
            json.dumps(row).replace("NaN", "NaN") + "\n" for row in rows
        ),
        encoding="utf-8",
    )

    decision = case.build()

    assert decision["training"]["all_finite"] is False
    assert (
        decision["acceptance_conditions"][
            "condition_01_fifty_applied_steps_no_skip_no_nonfinite"
        ]
        is False
    )


@pytest.mark.parametrize("clipped_steps", [10, 12, 25])
def test_condition_03_rejects_excessive_gradient_clipping(case, clipped_steps):
    case.write_run(ledger_rows=_ledger_rows(clipped_steps=clipped_steps))

    decision = case.build()

    assert decision["status"] == "FAIL-DIAGNOSED"
    assert (
        decision["acceptance_conditions"][
            "condition_03_clip_fraction_below_20_percent_no_streak_of_ten"
        ]
        is False
    )
    assert decision["training"]["longest_clipping_streak"] == clipped_steps


def test_condition_03_accepts_a_short_low_clipping_run(case):
    case.write_run(ledger_rows=_ledger_rows(clipped_steps=9))

    decision = case.build()

    assert (
        decision["acceptance_conditions"][
            "condition_03_clip_fraction_below_20_percent_no_streak_of_ten"
        ]
        is True
    )
    assert decision["training"]["clip_fraction"] == pytest.approx(0.18)


@pytest.mark.parametrize(
    ("condition", "field", "value"),
    [
        ("condition_04_uncond_raw_loss_dropped_at_least_10_percent", "uncond", 0.28),
        ("condition_06_common_idm_raw_loss_within_5_percent", "idm", 0.010),
        ("condition_07_gt_idm_action_l2_within_5_percent", "gt", 0.09),
        ("condition_08_generated_idm_action_l2_within_5_percent", "valid", 0.9),
        ("condition_09_sensitivity_median_retains_half", "sensitivity", 0.01),
    ],
)
def test_step50_metric_regressions_fail_closed(case, condition, field, value):
    case.write_run(diagnostic_overrides={50: {field: value}})

    decision = case.build()

    assert decision["status"] == "FAIL-DIAGNOSED"
    assert decision["acceptance_conditions"][condition] is False


def test_condition_05_requires_a_negative_uncond_slope(case):
    # UNCOND improves early then regresses: the endpoint drop still clears the
    # 10% gate, but the fitted slope over steps 0/10/25/50 is not negative.
    case.write_run(
        diagnostic_overrides={
            0: {"uncond": 0.20},
            10: {"uncond": 0.08},
            25: {"uncond": 0.08},
            50: {"uncond": 0.17},
        }
    )

    decision = case.build()

    assert (
        decision["acceptance_conditions"][
            "condition_04_uncond_raw_loss_dropped_at_least_10_percent"
        ]
        is True
    )
    assert (
        decision["acceptance_conditions"][
            "condition_05_uncond_raw_loss_linear_slope_is_negative"
        ]
        is False
    )
    assert decision["status"] == "FAIL-DIAGNOSED"


@pytest.mark.parametrize(
    ("condition", "coupling", "statistics"),
    [
        (
            "condition_10_common_action_all_margins_positive",
            "common",
            (0.052151287778994644, 5.526134393182538, -3.0),
        ),
        (
            "condition_11_independent_action_all_margins_positive",
            "independent",
            (0.03, 4.0, -2.0),
        ),
        # A far larger UNCOND norm pushes r_g past 1.25 while margins stay positive.
        (
            "condition_14_common_action_all_weighted_norm_ratio_at_most_1_25",
            "common",
            (0.05, 60.0, 1.0),
        ),
    ],
)
def test_aggregate_margin_regressions_fail_closed(case, condition, coupling, statistics):
    path = case.diagnostics[50] / "gradient_diagnostics.json"
    gradient = json.loads(path.read_text(encoding="utf-8"))
    gradient[coupling]["groups"]["action_all"] = _group(*statistics)
    path.write_text(json.dumps(gradient), encoding="utf-8")

    decision = case.build()

    assert decision["status"] == "FAIL-DIAGNOSED"
    assert decision["acceptance_conditions"][condition] is False


@pytest.mark.parametrize(
    ("condition", "group", "shards"),
    [
        (
            "condition_12_common_action_all_negative_shard_fractions_at_most_20_percent",
            "action_all",
            [(0.01, 5.5, -0.3)] * 4 + [(0.05, 5.5, -0.3)] * 4,
        ),
        (
            "condition_13_common_final_block_negative_idm_fraction_at_most_20_percent",
            "action_blocks_final",
            [(0.001, 1.0, -0.05)] * 4 + [(0.01, 1.0, -0.05)] * 4,
        ),
    ],
)
def test_shard_margin_regressions_fail_closed(case, condition, group, shards):
    path = case.diagnostics[50] / "gradient_diagnostics.json"
    gradient = json.loads(path.read_text(encoding="utf-8"))
    for shard, statistics in zip(gradient["common"]["shards"], shards):
        shard["groups"][group] = _group(*statistics)
    path.write_text(json.dumps(gradient), encoding="utf-8")

    decision = case.build()

    assert decision["status"] == "FAIL-DIAGNOSED"
    assert decision["acceptance_conditions"][condition] is False


@pytest.mark.parametrize("step", [0, 10, 25, 50])
def test_condition_15_rejects_no_read_parity_above_tolerance(case, step):
    path = case.diagnostics[step] / "generated_future_validation.json"
    generated = json.loads(path.read_text(encoding="utf-8"))
    generated["no_read_uncond_parity"]["max_abs"] = 2e-4
    path.write_text(json.dumps(generated), encoding="utf-8")

    decision = case.build()

    assert (
        decision["acceptance_conditions"][
            "condition_15_no_read_forced_uncond_parity_within_1e_4"
        ]
        is False
    )
    assert decision["status"] == "FAIL-DIAGNOSED"


def test_condition_16_rejects_a_bound_artifact_whose_bytes_changed(case):
    case.files["dataset_stats"].write_text("tampered", encoding="utf-8")

    decision = case.build()

    assert (
        decision["acceptance_conditions"][
            "condition_16_all_bound_artifact_sha256_reverified"
        ]
        is False
    )
    assert any(
        "dataset_stats" in reason
        for reason in decision["artifact_reverification"]["failures"]
    )


def test_condition_16_rejects_an_unverifiable_code_commit(case):
    (case.repo / "dirty.txt").write_text("uncommitted", encoding="utf-8")

    decision = case.build()

    assert (
        decision["acceptance_conditions"][
            "condition_16_all_bound_artifact_sha256_reverified"
        ]
        is False
    )
    assert (
        decision["artifact_reverification"]["artifacts"]["code_commit"][
            "worktree_dirty"
        ]
        is True
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data.train.manifest_split", "all"),
        ("data.train.manifest_split", None),
        ("data.train.episode_split_manifest", None),
    ],
)
def test_condition_17_requires_validation_episodes_to_stay_excluded(case, field, value):
    case.write_run(config_overrides={field: value})

    decision = case.build()

    assert (
        decision["acceptance_conditions"][
            "condition_17_validation_episodes_excluded_from_training_sampler"
        ]
        is False
    )
    assert decision["validation_exclusion"]["reasons"]


def test_condition_17_rejects_a_split_manifest_that_is_not_the_bound_one(case):
    other = case.artifacts / "other_manifest.json"
    other.write_text("{}", encoding="utf-8")
    case.write_run(config_overrides={"data.train.episode_split_manifest": str(other)})

    decision = case.build()

    assert (
        decision["acceptance_conditions"][
            "condition_17_validation_episodes_excluded_from_training_sampler"
        ]
        is False
    )


# ---------------------------------------------------------------------------
# Plan section 11, scenario 1: reject a trigger that is not FAIL-DIAGNOSED
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", ["PASS", "NOT-RUN", "FAIL", "fail-diagnosed"])
def test_rejects_failed_canary_that_is_not_fail_diagnosed(case, status):
    case.write_failed_canary(status=status)

    with pytest.raises(ValueError, match="FAIL-DIAGNOSED"):
        case.build()

    result = case.run_cli()
    assert result["status"] == "NOT-RUN"
    assert result["probe_500_authorized"] is False


def test_rejects_trigger_without_the_common_noise_margin_failure(case):
    case.write_failed_canary(failure_conditions=["gradient clipping streak"])

    with pytest.raises(ValueError, match="common-noise IDM margin is not positive"):
        case.build()


def test_rejects_trigger_with_the_wrong_schema(case):
    case.write_failed_canary(schema="fastwam-sdr-learning-probe-decision-v1")

    with pytest.raises(ValueError, match="fastwam-sdr-canary-decision-v1"):
        case.build()


# ---------------------------------------------------------------------------
# Plan section 11, scenario 2: reject a changed failed-Canary SHA256
# ---------------------------------------------------------------------------
def test_rejects_changed_failed_canary_decision_sha256(case):
    pinned = _sha256(case.failed_path)
    payload = json.loads(case.failed_path.read_text(encoding="utf-8"))
    payload["probe_500_authorized"] = True
    case.failed_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _sha256(case.failed_path) != pinned

    with pytest.raises(ValueError, match="Failed Canary decision changed"):
        case.build("--expected-failed-canary-sha256", pinned)


def test_archived_failed_canary_matches_the_pinned_trigger_sha256():
    assert ARCHIVED_CANARY.is_file()
    payload = json.loads(ARCHIVED_CANARY.read_text(encoding="utf-8"))

    result = validate_failed_canary_trigger(
        payload, observed_sha256=_sha256(ARCHIVED_CANARY)
    )

    assert _sha256(ARCHIVED_CANARY) == ARCHIVED_FAILED_CANARY_DECISION_SHA256
    assert result["status"] == "FAIL-DIAGNOSED"
    assert "common-noise IDM margin is not positive" in result["failure_conditions"]


def test_trigger_validation_requires_an_observed_sha_when_pinned():
    payload = json.loads(ARCHIVED_CANARY.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="SHA256 must be supplied"):
        validate_failed_canary_trigger(payload)


# ---------------------------------------------------------------------------
# Plan section 11, scenario 3: reject E-I/config/stats/manifest/solver drift
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(LOW_CAP_INVARIANT_BINDING_NAMES))
def test_rejects_invariant_artifact_mismatch_against_the_failed_canary(case, name):
    bindings = copy.deepcopy(case.bindings)
    bindings[name] = {"path": bindings[name]["path"], "sha256": "f" * 64}
    case.write_failed_canary(artifact_bindings=bindings)

    with pytest.raises(ValueError, match="must reuse the failed Canary artifacts"):
        case.build()


@pytest.mark.parametrize("name", list(LOW_CAP_INVARIANT_BINDING_NAMES))
def test_rejects_a_missing_invariant_binding(case, name):
    bindings = copy.deepcopy(case.bindings)
    del bindings[name]
    case.write_preflight(artifact_bindings=bindings)

    with pytest.raises(ValueError, match="missing invariant artifact bindings"):
        case.build()


def test_invariant_binding_helper_accepts_only_identical_shas():
    fresh = {
        name: char * 64
        for name, char in zip(LOW_CAP_INVARIANT_BINDING_NAMES, "abcdefgh")
    }
    assert validate_low_cap_invariant_bindings(fresh, dict(fresh)) == fresh

    drifted = dict(fresh)
    drifted["e_i_checkpoint"] = "9" * 64
    with pytest.raises(ValueError, match="must reuse the failed Canary artifacts"):
        validate_low_cap_invariant_bindings(fresh, drifted)


# ---------------------------------------------------------------------------
# Plan section 11, scenario 4: reject a cap that is not exactly 0.09
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cap", [0.5, 0.2, 0.1, 0.095, 0.0900001])
def test_rejects_a_cap_above_the_locked_value(cap):
    schedule = [[0.0, W0], [0.1, W0], [0.3, min(0.2, cap)], [0.6, cap], [1.0, cap]]

    with pytest.raises(ValueError, match="exceed the float32 cap"):
        validate_low_cap_uncond_weight_schedule(schedule)


@pytest.mark.parametrize("cap", [0.089, 0.08999999, 0.05, 0.01])
def test_rejects_a_cap_below_the_locked_value(cap):
    schedule = [[0.0, W0], [0.1, W0], [0.3, cap], [0.6, cap], [1.0, cap]]

    with pytest.raises(ValueError, match="w_cap must be exactly 0.09"):
        validate_low_cap_uncond_weight_schedule(schedule)


def test_accepts_only_the_locked_cap():
    result = validate_low_cap_uncond_weight_schedule(low_cap_locked_schedule(W0))

    assert result["w_cap"] == 0.09
    assert result["w0"] == W0
    assert result["w_cap_float32"] == float32(0.09)


@pytest.mark.parametrize(
    "fractions", [(0.0, 0.2, 0.3, 0.6, 1.0), (0.0, 0.1, 0.3, 0.6, 0.9)]
)
def test_rejects_schedule_fractions_that_are_not_preregistered(fractions):
    schedule = [
        [fraction, weight]
        for fraction, weight in zip(fractions, [W0, W0, 0.09, 0.09, 0.09])
    ]

    with pytest.raises(ValueError, match="fractions must be"):
        validate_low_cap_uncond_weight_schedule(schedule)


def test_resolved_config_schedule_must_serialize_the_locked_schedule(case):
    case.write_run(
        config_overrides={
            "dual_regime_training.uncond_weight_schedule": [
                [0.0, W0],
                [0.1, W0],
                [0.3, 0.2],
                [0.6, 0.5],
                [1.0, 0.5],
            ]
        }
    )

    with pytest.raises(RuntimeError, match="does not serialize the locked schedule"):
        case.build()


# ---------------------------------------------------------------------------
# Plan section 11, scenario 5: reject any schedule point above the cap
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_rejects_a_schedule_point_above_the_float32_cap(index):
    schedule = low_cap_locked_schedule(W0)
    schedule[index][1] = 0.0900001

    with pytest.raises(ValueError, match="exceed the float32 cap"):
        validate_low_cap_uncond_weight_schedule(schedule)


def test_rejects_a_logged_weight_above_the_float32_cap(case):
    rows = _ledger_rows()
    rows[30]["losses"]["action_regime_weight_uncond"] = float32(0.0900001)
    _write_jsonl(case.training_metrics, rows)

    decision = case.build()

    assert (
        decision["acceptance_conditions"][
            "condition_02_weight_matches_locked_schedule_and_never_exceeds_cap"
        ]
        is False
    )
    assert decision["schedule"]["per_step_verification"]["cap_exceedances"] == [
        {"global_step": 31, "observed": float32(0.0900001)}
    ]
    assert decision["status"] == "FAIL-DIAGNOSED"


def test_rejects_a_logged_weight_that_drifts_from_the_locked_schedule(case):
    rows = _ledger_rows()
    rows[7]["losses"]["action_regime_weight_uncond"] = float32(0.02)
    _write_jsonl(case.training_metrics, rows)

    decision = case.build()

    assert (
        decision["acceptance_conditions"][
            "condition_02_weight_matches_locked_schedule_and_never_exceeds_cap"
        ]
        is False
    )
    assert decision["schedule"]["per_step_verification"]["worst_absolute_error"] > 0.0


def test_rejects_a_ledger_produced_by_the_old_high_cap_schedule(case):
    _write_jsonl(case.training_metrics, _ledger_rows(cap=0.5))

    decision = case.build()

    assert (
        decision["acceptance_conditions"][
            "condition_02_weight_matches_locked_schedule_and_never_exceeds_cap"
        ]
        is False
    )
    assert decision["schedule"]["per_step_verification"]["cap_exceedances"]


# ---------------------------------------------------------------------------
# Plan section 11, scenario 6: reject changed steps, seed, LR, trainable groups
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 100),
        ("seed", 1234),
        ("learning_rate", 3e-5),
        ("dual_regime_training.optimizer.action_lr_scale", 0.5),
        ("dual_regime_training.optimizer.proprio_lr_scale", 1.0),
        ("dual_regime_training.optimizer.video_lr_scale", 1.0),
        ("gradient_accumulation_steps", 32),
        ("batch_size", 2),
        ("mixed_precision", "fp16"),
        ("max_grad_norm", 5.0),
        ("weight_decay", 0.0),
        ("lr_scheduler_type", "constant"),
        ("weights_checkpoint_kind", "full"),
    ],
)
def test_rejects_a_changed_non_cap_training_setting(case, field, value):
    case.write_run(config_overrides={field: value})

    decision = case.build()

    assert decision["status"] == "FAIL-DIAGNOSED"
    assert decision["single_variable"]["audit"]["pass"] is False
    assert any(
        "non-w_cap training settings changed" in reason
        for reason in decision["contract_violations"]
    )


def test_rejects_a_per_step_learning_rate_that_differs_from_the_failed_canary(case):
    rows = _ledger_rows()
    rows[12]["learning_rate"] = 2e-5
    _write_jsonl(case.training_metrics, rows)

    decision = case.build()

    assert decision["status"] == "FAIL-DIAGNOSED"
    assert decision["learning_rate_reference"]["matches_reference"] is False
    assert (
        "per-step learning rate differs from the failed Canary reference"
        in decision["contract_violations"]
    )


def test_accepts_a_learning_rate_sequence_identical_to_the_failed_canary(case):
    decision = case.build()

    assert decision["learning_rate_reference"]["matches_reference"] is True
    assert decision["learning_rate_reference"]["worst_absolute_difference"] == 0.0
    assert decision["learning_rate_reference"]["reference_step_count"] == 50


def test_rejects_a_fresh_w0_that_drifted_more_than_five_percent(case):
    case.write_preflight(w0=W0 * 1.2)

    with pytest.raises(ValueError, match="drifted from the archived value"):
        case.build()


@pytest.mark.parametrize("w0", [0.0009, 0.06, 0.0, -0.01])
def test_rejects_a_fresh_w0_outside_the_preregistered_range(case, w0):
    # The message is asserted exactly: a loose match here would be satisfied by
    # the separate 5% drift gate and would leave the range gate unprotected.
    case.write_preflight(w0=w0)

    with pytest.raises(ValueError, match="outside the low-cap range"):
        case.build()


@pytest.mark.parametrize("w0", [0.0009, 0.06, 0.0, -0.01, 1.0])
def test_locked_schedule_builder_rejects_an_out_of_range_w0(w0):
    with pytest.raises(ValueError, match=r"w0 must satisfy"):
        low_cap_locked_schedule(w0)


def test_accepts_a_fresh_w0_inside_the_five_percent_band(case):
    case.w0 = W0 * 1.04
    case.write_preflight()
    case.write_run(ledger_rows=_ledger_rows(w0=case.w0))
    case.preregister()

    decision = case.build()

    assert decision["status"] == "PASS"
    assert decision["fresh_preflight"]["w0_relative_drift"] == pytest.approx(0.04)
    assert decision["schedule"]["locked"][0][1] == pytest.approx(W0 * 1.04)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_seed", 7),
        ("diagnostic_seed", 1),
        ("max_steps", 500),
        ("learning_rate", 3e-5),
        ("new_w_cap", 0.5),
        ("old_w_cap", 0.09),
        ("stage", "E1-P1"),
    ],
)
def test_rejects_a_preregistration_that_changed_a_frozen_field(case, field, value):
    payload = json.loads(case.preregistration_path.read_text(encoding="utf-8"))
    payload[field] = value
    case.preregistration_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=field):
        case.build()


# ---------------------------------------------------------------------------
# Plan section 11, scenario 7: the original probe route is unchanged
# ---------------------------------------------------------------------------
def test_original_learning_probe_contract_still_rejects_the_low_cap_value():
    preflight = {
        "schema": PREFLIGHT_DECISION_SCHEMA,
        "status": "PASS",
        "plus_full_used": False,
        "artifact_bindings": {"e_i_checkpoint": {"path": "/x", "sha256": "b" * 64}},
        "w0": 0.05,
        "w_cap": 0.09,
        "allowed_schedule": [
            [0.0, 0.05],
            [0.1, 0.05],
            [0.3, 0.09],
            [0.6, 0.09],
            [1.0, 0.09],
        ],
    }

    with pytest.raises(ValueError, match="outside the allowed range"):
        validate_learning_probe_contract(preflight)

    preflight["w_cap"] = 0.5
    preflight["allowed_schedule"] = [
        [0.0, 0.05],
        [0.1, 0.05],
        [0.3, 0.2],
        [0.6, 0.5],
        [1.0, 0.5],
    ]
    assert validate_learning_probe_contract(preflight)["w_cap"] == 0.5


def test_original_learning_probe_launcher_is_untouched():
    launcher = PROBE_LAUNCHER.read_text(encoding="utf-8")

    assert "exec bash scripts/adaptive_gate/run_e1_sdr_formal_train.sh" in launcher
    assert "decide_sdr_learning_probe.py canary" in launcher
    assert "decide_sdr_learning_probe.py probe" in launcher
    assert "low_cap" not in launcher
    assert "0.09" not in launcher


def test_original_canary_decision_schema_is_reused_not_rewritten():
    assert CANARY_DECISION_SCHEMA == "fastwam-sdr-canary-decision-v1"
    assert LOW_CAP_CANARY_DECISION_SCHEMA != CANARY_DECISION_SCHEMA
    archived = json.loads(ARCHIVED_CANARY.read_text(encoding="utf-8"))
    assert archived["schema"] == CANARY_DECISION_SCHEMA
    assert archived["status"] == "FAIL-DIAGNOSED"
    assert archived["probe_500_authorized"] is False


# ---------------------------------------------------------------------------
# Plan section 11, scenario 8: a PASS never chains into 500-step or formal
# ---------------------------------------------------------------------------
def test_low_cap_launcher_never_invokes_the_probe_or_formal_launcher():
    launcher = LOW_CAP_LAUNCHER.read_text(encoding="utf-8")

    assert "run_e1_sdr_formal_train.sh" not in launcher
    assert "run_e1_sdr_learning_probe.sh" not in launcher
    assert "decide_sdr_learning_probe.py" not in launcher
    assert "exec " not in launcher
    assert "SDR_LEARNING_PROBE_DECISION" not in launcher
    assert "max_steps=50" in launcher
    assert "max_steps=500" not in launcher


def test_low_cap_pass_does_not_authorize_formal_training(case):
    decision = case.build()

    assert decision["status"] == "PASS"
    assert decision["formal_training_authorized"] is False
    assert decision["probe_500_authorized"] is True
    assert "preregister" in decision["next_stage"]


def test_low_cap_launcher_saves_the_three_required_deltas_and_stops():
    launcher = LOW_CAP_LAUNCHER.read_text(encoding="utf-8")

    assert "save_steps=[10,25,50]" in launcher
    assert "for step in 10 25 50; do" in launcher
    assert "warm_start.kind=standalone_idm" in launcher
    assert 'E_I_SHA256=$(file_sha256 "${E_I_CKPT}")' in launcher
    assert '"warm_start.expected_checkpoint_sha256=${E_I_SHA256}"' in launcher
    assert "decide_sdr_low_cap_canary.py" in launcher
    assert launcher.rstrip().endswith(
        "separate 500-step low-cap probe as its own experiment."
    )


def test_low_cap_launcher_requires_the_documented_environment_contract():
    launcher = LOW_CAP_LAUNCHER.read_text(encoding="utf-8")

    for name in (
        "E_I_BASE_MODEL_MANIFEST",
        "E_I_CKPT",
        "E_I_CONFIG",
        "E_I_LINEAGE_MANIFEST",
        "DATASET_STATS",
        "WARMSTART_DECISION",
        "SDR_VAL_MANIFEST",
        "GENERATED_FUTURE_CACHE_SOURCE",
        "FAILED_CANARY_DECISION",
        "FRESH_SDR_PREFLIGHT_DECISION",
        "LOW_CAP_PLAN",
    ):
        assert name in launcher
    assert "parse_launcher_args" in launcher


# ---------------------------------------------------------------------------
# Plan section 11, scenario 9: no re-selection of the step 10/25 checkpoints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("step", [10, 25, 0, 49])
def test_decision_cannot_reselect_an_intermediate_checkpoint(case, step):
    with pytest.raises(RuntimeError, match="re-selecting step"):
        case.build("--selected-step", str(step))

    result = case.run_cli("--selected-step", str(step))
    assert result["status"] == "NOT-RUN"
    assert result["selected_checkpoint_step"] == 50


def test_decision_always_records_step_50_as_the_only_outcome(case):
    decision = case.build()

    assert decision["selected_checkpoint_step"] == 50
    assert decision["intermediate_step_reselection"] == "prohibited"
    assert set(decision["checkpoint_deltas"]) == {"step10", "step25", "step50"}
    assert set(decision["evaluation_steps"]) == {"0", "10", "25", "50"}


def test_a_healthy_step_10_cannot_rescue_a_failing_step_50(case):
    case.write_run(diagnostic_overrides={50: {"valid": 0.9}})

    decision = case.build()

    assert decision["status"] == "FAIL-DIAGNOSED"
    assert (
        decision["acceptance_conditions"][
            "condition_08_generated_idm_action_l2_within_5_percent"
        ]
        is False
    )
    assert decision["evaluation_steps"]["10"]["generated_idm_action_l2"] == pytest.approx(
        STEP_PROFILE[10]["valid"]
    )
    assert decision["selected_checkpoint_step"] == 50


# ---------------------------------------------------------------------------
# Plan section 11, scenario 10: Plus-Full must be absent or explicitly false
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", [True, None, "false", 0])
def test_rejects_a_preflight_whose_plus_full_flag_is_not_false(case, value):
    payload = json.loads(case.preflight_path.read_text(encoding="utf-8"))
    if value is None:
        del payload["plus_full_used"]
    else:
        payload["plus_full_used"] = value
    case.preflight_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="plus_full_used=false"):
        case.build()


@pytest.mark.parametrize("value", [True, None])
def test_rejects_a_failed_canary_whose_plus_full_flag_is_not_false(case, value):
    payload = json.loads(case.failed_path.read_text(encoding="utf-8"))
    if value is None:
        del payload["plus_full_used"]
    else:
        payload["plus_full_used"] = value
    case.failed_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="plus_full_used=false"):
        case.build()


def test_rejects_a_preregistration_that_claims_plus_full_was_used(case):
    payload = json.loads(case.preregistration_path.read_text(encoding="utf-8"))
    payload["plus_full_used"] = True
    case.preregistration_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="plus_full_used"):
        case.build()


@pytest.mark.parametrize(
    ("binding_name", "file_name"),
    [
        ("libero_plus_manifest", "manifest.json"),
        ("headline_manifest", "libero_plus_full_manifest.json"),
    ],
)
def test_condition_18_rejects_a_plus_full_artifact_binding(case, binding_name, file_name):
    bindings = copy.deepcopy(case.bindings)
    plus_path = case.artifacts / file_name
    plus_path.write_text("{}", encoding="utf-8")
    bindings[binding_name] = {"path": str(plus_path), "sha256": _sha256(plus_path)}
    case.write_preflight(artifact_bindings=bindings)

    decision = case.build()

    assert (
        decision["acceptance_conditions"][
            "condition_18_plus_full_outcome_not_loaded_or_used"
        ]
        is False
    )
    assert decision["status"] == "FAIL-DIAGNOSED"


def test_decision_always_states_plus_full_was_not_used(case):
    decision = case.build()

    assert decision["plus_full_used"] is False
    assert "No Plus-Full" in decision["no_plus_full_statement"]


# ---------------------------------------------------------------------------
# Provenance failures become NOT-RUN, never PASS
# ---------------------------------------------------------------------------
def test_recomputation_cross_check_rejects_tampered_persisted_margins(case):
    path = case.diagnostics[50] / "gradient_diagnostics.json"
    gradient = json.loads(path.read_text(encoding="utf-8"))
    gradient["common"]["groups"]["action_all"]["margins_by_weight"]["0.1"][
        "idm_margin"
    ] = 1.0
    path.write_text(json.dumps(gradient), encoding="utf-8")

    with pytest.raises(RuntimeError, match="disagrees with the persisted grid"):
        case.build()


def test_reference_learning_rate_ledger_must_match_its_recorded_sha(case):
    case.reference_ledger.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Reference Canary training ledger changed"):
        case.build()


def test_missing_diagnostic_directory_is_not_run(case):
    shutil.rmtree(case.diagnostics[25])

    result = case.run_cli()

    assert result["status"] == "NOT-RUN"
    assert result["acceptance_conditions"] == {}
    assert result["probe_500_authorized"] is False
    assert result["formal_training_authorized"] is False


def test_preregistration_freezes_the_locked_schedule_and_trigger(case):
    payload = json.loads(case.preregistration_path.read_text(encoding="utf-8"))

    assert payload["schema"] == LOW_CAP_PREREGISTRATION_SCHEMA
    assert payload["stage"] == "E1-P1D-LC"
    assert payload["old_w_cap"] == 0.5
    assert payload["new_w_cap"] == 0.09
    assert payload["learning_rate"] == 1e-5
    assert payload["max_steps"] == 50
    assert payload["training_seed"] == 42
    assert payload["diagnostic_seed"] == 20260721
    assert payload["plus_full_used"] is False
    assert payload["locked_schedule"] == low_cap_locked_schedule(W0)
    assert payload["delta_steps"] == [10, 25, 50]
    assert payload["trigger"]["status"] == "FAIL-DIAGNOSED"


def test_evidence_index_is_written_without_asserting_an_interpretation(case):
    decision = case.build()
    case.decision_path.write_text(json.dumps(decision), encoding="utf-8")
    args = MODULE.parser().parse_args(
        [
            "write-evidence-index",
            "--decision",
            str(case.decision_path),
            "--run-dir",
            str(case.run_dir),
        ]
    )
    MODULE.write_evidence_index(args)

    sums = (case.run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8")
    readme = (case.run_dir / "README.md").read_text(encoding="utf-8")

    assert "low_cap_canary_decision.json" in sums
    assert _sha256(case.decision_path) in sums
    assert "Machine-generated index" in readme
    assert "decision status: `PASS`" in readme
    assert "w_cap" in readme


# ---------------------------------------------------------------------------
# Launcher dry-run: exact ordered ledger, no decision artifact
# ---------------------------------------------------------------------------
def _launcher_harness(tmp_path: Path) -> Path:
    """Mirror the two-repo layout the shared launcher _common.sh expects."""
    outer_tools = PROJECT_ROOT.parent.parent / "scripts/adaptive_gate"
    if not (outer_tools / "_common.sh").is_file():
        pytest.skip("outer scripts/adaptive_gate is unavailable")
    workspace = tmp_path / "workspace"
    (workspace / "scripts").mkdir(parents=True, exist_ok=True)
    os.symlink(outer_tools, workspace / "scripts/adaptive_gate")
    project = workspace / "FastWAM"
    project.mkdir(parents=True, exist_ok=True)
    for name in ("scripts", "configs", "src", "docs"):
        os.symlink(PROJECT_ROOT / name, project / name)
    return project


def test_launcher_dry_run_emits_the_ordered_ledger_without_a_decision(tmp_path):
    project = _launcher_harness(tmp_path)
    baseline = tmp_path / "fresh_preflight"
    baseline.mkdir(parents=True, exist_ok=True)
    (baseline / "resolved_config.yaml").write_text(
        "placeholder: true\n", encoding="utf-8"
    )
    (baseline / "generated_future_validation.json").write_text(
        json.dumps(
            {
                "schema": "fastwam-sdr-generated-future-validation-v1",
                "status": "PASS",
                "cache_created": True,
                "direct_generated_cached_valid_parity": {"pass": True},
            }
        ),
        encoding="utf-8",
    )
    # Reuse the archived trigger so the pinned default SHA256 is exercised.
    failed = tmp_path / "canary_decision.json"
    shutil.copyfile(ARCHIVED_CANARY, failed)
    archived_bindings = json.loads(failed.read_text(encoding="utf-8"))[
        "artifact_bindings"
    ]
    (baseline / "preflight_decision.json").write_text(
        json.dumps(
            {
                "schema": PREFLIGHT_DECISION_SCHEMA,
                "status": "PASS",
                "plus_full_used": False,
                "artifact_bindings": archived_bindings,
                "w0": W0,
                "w_cap": 0.5,
                "allowed_schedule": [
                    [0.0, W0],
                    [0.1, W0],
                    [0.3, 0.2],
                    [0.6, 0.5],
                    [1.0, 0.5],
                ],
            }
        ),
        encoding="utf-8",
    )
    inputs = {}
    for name in (
        "E_I_BASE_MODEL_MANIFEST",
        "E_I_CKPT",
        "E_I_CONFIG",
        "E_I_LINEAGE_MANIFEST",
        "DATASET_STATS",
        "WARMSTART_DECISION",
        "SDR_VAL_MANIFEST",
        "LOW_CAP_PLAN",
    ):
        path = tmp_path / f"{name.lower()}.bin"
        path.write_text(name, encoding="utf-8")
        inputs[name] = str(path)

    env = {
        **os.environ,
        **inputs,
        "GENERATED_FUTURE_CACHE_SOURCE": str(baseline),
        "FAILED_CANARY_DECISION": str(failed),
        "FRESH_SDR_PREFLIGHT_DECISION": str(baseline / "preflight_decision.json"),
        "EXPERIMENT_ROOT": str(tmp_path / "experiments"),
        "RUN_ID": "20260726T000000Z",
        "PYTHONPATH": os.pathsep.join(
            [
                str(PROJECT_ROOT / "src"),
                str(PROJECT_ROOT),
                os.environ.get("PYTHONPATH", ""),
            ]
        ),
    }
    result = subprocess.run(
        ["bash", str(project / "scripts/adaptive_gate/run_e1_sdr_low_cap_canary.sh"), "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    joined = "\n".join(
        line
        for line in result.stdout.splitlines()
        if line.startswith("__ADAPTIVE_WM_COMMAND__")
    )
    order = [
        "sdr_stage_contract.py check-lineage",
        "sdr_stage_contract.py check-decision",
        "decide_sdr_low_cap_canary.py preregister",
        "train_zero1.sh",
        "run_sdr_preflight.py",
        "decide_sdr_low_cap_canary.py decide",
        "decide_sdr_low_cap_canary.py write-evidence-index",
    ]
    positions = [joined.find(token) for token in order]
    assert all(position >= 0 for position in positions), (order, positions, joined)
    assert positions == sorted(positions), joined
    assert joined.count("run_sdr_preflight.py") == 3
    unquoted = joined.replace("\\", "")
    for step in (10, 25, 50):
        assert f"step_{step:06d}.action_dit_delta.pt" in unquoted
    assert "save_steps=[10,25,50]" in unquoted
    assert "[0.3,0.09],[0.6,0.09],[1.0,0.09]" in unquoted
    assert "run_e1_sdr_formal_train.sh" not in joined
    assert not list((tmp_path / "experiments").rglob("*decision*.json"))
