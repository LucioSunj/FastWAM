"""Cross-arm table and GO/NO-GO decision for the BC dropout pilot."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from fastwam.modality_dropout_bc import (
    PILOT_ARMS,
    paired_bootstrap_reliance_change,
    random_patch_kill_test,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def load_arm_evidence(path: str | Path) -> dict[str, Any]:
    """Load one arm result plus its step-zero and endpoint paired arrays."""

    candidate = Path(path).expanduser().resolve()
    result_path = candidate / "arm_result.json" if candidate.is_dir() else candidate
    result = _read_json(result_path)
    if result.get("schema") != "fastwam-modality-dropout-arm-result-v1":
        raise ValueError(f"Unsupported arm-result schema in {result_path}.")
    arm = str(result["arm"]["name"])
    endpoint = int(result["global_step"])
    step_zero = _read_json(result_path.parent / "heldout_step_000000.json")
    final = _read_json(result_path.parent / f"heldout_step_{endpoint:06d}.json")
    for payload, step in ((step_zero, 0), (final, endpoint)):
        if payload.get("schema") != "fastwam-modality-dropout-heldout-v1":
            raise ValueError(f"Unsupported held-out schema for arm {arm}.")
        if payload.get("arm") != arm or int(payload.get("step", -1)) != step:
            raise ValueError(f"Held-out artifact identity mismatch for arm {arm}.")
    return {"result": result, "step_zero": step_zero, "final": final}


def _paired_mean_change(
    first: Sequence[float],
    second: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    left = torch.as_tensor(first, dtype=torch.float64)
    right = torch.as_tensor(second, dtype=torch.float64)
    if left.ndim != 1 or left.shape != right.shape or left.numel() < 2:
        raise ValueError(
            "Paired mean-change arrays must be aligned one-dimensional data."
        )
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError("Paired mean-change arrays contain non-finite values.")
    difference = right - left
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    samples = []
    remaining = int(draws)
    while remaining:
        count = min(remaining, 512)
        index = torch.randint(
            difference.numel(),
            (count, difference.numel()),
            generator=generator,
        )
        samples.append(difference[index].mean(dim=1))
        remaining -= count
    bootstrap = torch.cat(samples)
    ci = torch.quantile(
        bootstrap,
        torch.tensor([0.025, 0.975], dtype=bootstrap.dtype),
    ).tolist()
    return {
        "point": float(difference.mean()),
        "ci95": [float(ci[0]), float(ci[1])],
        "paired_samples": int(difference.numel()),
        "bootstrap_draws": int(draws),
        "seed": int(seed),
    }


def _rollout_summary(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"status": "NOT-RUN", "arms": {}}
    if payload.get("schema") != "fastwam-modality-dropout-rollout-v1":
        raise ValueError("Rollout evidence has an unsupported schema.")
    arms = payload.get("arms")
    if not isinstance(arms, Mapping):
        raise TypeError("Rollout evidence must contain an arm mapping.")
    result: dict[str, Any] = {"status": "COMPLETE", "arms": {}}
    for arm, conditions in arms.items():
        if not isinstance(conditions, Mapping):
            raise TypeError(f"Rollout conditions for arm {arm} must be a mapping.")
        condition_result = {}
        for condition in ("clean", "wan_drop", "dino_drop"):
            values = conditions.get(condition, {}).get("episode_success")
            if not isinstance(values, list) or len(values) != 30:
                raise ValueError(
                    f"Rollout arm {arm} condition {condition} requires 30 outcomes."
                )
            tensor = torch.as_tensor(values, dtype=torch.float64)
            condition_result[condition] = {
                "episodes": 30,
                "success_rate": float(tensor.mean()),
            }
        condition_result["d_dino_sr"] = (
            condition_result["clean"]["success_rate"]
            - condition_result["dino_drop"]["success_rate"]
        )
        condition_result["d_wan_sr"] = (
            condition_result["clean"]["success_rate"]
            - condition_result["wan_drop"]["success_rate"]
        )
        result["arms"][str(arm)] = condition_result
    return result


def _language_summary(payload: Mapping[str, Any] | None) -> dict[str, Any] | str:
    """Validate the lightweight paired-instruction canary artifact."""

    if payload is None:
        return "NOT-RUN"
    if payload.get("schema") != "fastwam-modality-dropout-language-canary-v1":
        raise ValueError("Language-canary evidence has an unsupported schema.")
    if payload.get("status") != "COMPLETE":
        return dict(payload)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not 10 <= len(cases) <= 14:
        raise ValueError("A complete language canary requires about 12 listed cases.")
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise TypeError(f"Language-canary case {index} must be a mapping.")
        first = str(case.get("instruction_a", "")).strip()
        second = str(case.get("instruction_b", "")).strip()
        if not first or not second or first == second:
            raise ValueError(
                f"Language-canary case {index} requires two distinct instructions."
            )
        if "behaviors_differ" not in case:
            raise ValueError(
                f"Language-canary case {index} lacks `behaviors_differ`."
            )
    if not isinstance(payload.get("material_degradation"), bool):
        raise TypeError("Complete language canary needs bool material_degradation.")
    return dict(payload)


def _mechanism_diagnosis(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
) -> dict[str, Any]:
    """Identify gate/projection/residual divergence without hidden thresholds."""

    before = initial["diagnostics"]["overall"]
    after = final["diagnostics"]["overall"]
    change = {
        name: float(after[name]) - float(before[name])
        for name in (
            "gate_mean",
            "projected_norm",
            "residual_norm",
            "residual_cross_sample_variance",
        )
    }
    transfer = (
        change["gate_mean"] > 0.0
        and change["projected_norm"] < 0.0
        and change["residual_norm"] <= 0.0
    )
    directions = {
        name: "up" if value > 0.0 else "down" if value < 0.0 else "flat"
        for name, value in change.items()
    }
    return {
        "change": change,
        "direction": directions,
        "collapse_transfer_pattern": transfer,
        "interpretation": (
            "upstream_cancellation"
            if transfer
            else "no_gate_projection_residual_divergence"
        ),
        "amplitude_collapse": (
            "DESCRIPTIVE_TOWARD_ZERO"
            if change["residual_norm"] < 0.0
            else "NO_DECREASE_OBSERVED"
        ),
        "constant_collapse": (
            "DESCRIPTIVE_TOWARD_CONSTANT"
            if change["residual_cross_sample_variance"] < 0.0
            and change["residual_norm"] >= 0.0
            else "NO_VARIANCE_NORM_DIVERGENCE_OBSERVED"
        ),
        "decision_use": "MECHANISM_DIAGNOSTIC_ONLY",
    }


def decide_modality_dropout_pilot(
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    rollout: Mapping[str, Any] | None = None,
    language_canary: Mapping[str, Any] | None = None,
    ood: Mapping[str, Any] | None = None,
    bootstrap_draws: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Build the preregistered result table and one conservative decision."""

    if set(evidence) != set(PILOT_ARMS):
        raise ValueError(
            f"Decision requires exactly the six arms {sorted(PILOT_ARMS)}."
        )
    endpoints = {
        int(payload["result"]["global_step"]) for payload in evidence.values()
    }
    initialization = {
        json.dumps(payload["result"]["initial_trainable_sha256"], sort_keys=True)
        for payload in evidence.values()
    }
    frozen_initialization = {
        payload["result"].get("initial_frozen_parent_and_dino_sha256")
        for payload in evidence.values()
    }
    heldout_indices = {
        json.dumps(payload["result"].get("heldout_indices"), sort_keys=True)
        for payload in evidence.values()
    }
    heldout_contract_ok = all(
        isinstance(payload["result"].get("heldout_indices"), list)
        and len(payload["result"]["heldout_indices"]) == 256
        and len(set(payload["result"]["heldout_indices"])) == 256
        and payload["result"].get("heldout_split_contract")
        == "episode-disjoint-from-training"
        for payload in evidence.values()
    )
    heldout_arrays_ok = all(
        all(
            isinstance(payload[checkpoint]["losses"].get(condition), list)
            and len(payload[checkpoint]["losses"][condition]) == 256
            for checkpoint in ("step_zero", "final")
            for condition in ("clean", "wan_drop", "dino_drop", "both_drop")
        )
        for payload in evidence.values()
    )
    arm_identity_ok = all(
        payload["result"].get("arm", {}).get("name") == arm
        and payload["step_zero"].get("arm") == arm
        and payload["final"].get("arm") == arm
        for arm, payload in evidence.items()
    )
    frozen_unchanged = all(
        payload["result"].get("initial_frozen_parent_and_dino_sha256")
        == payload["result"].get("final_frozen_parent_and_dino_sha256")
        and payload["result"].get("frozen_parameter_versions_unchanged") is True
        for payload in evidence.values()
    )
    all_arms_complete = all(
        payload["result"].get("status") == "COMPLETE"
        for payload in evidence.values()
    )
    baseline_platformed = (
        evidence["A"]["result"]["endpoint"].get("status") == "PLATFORMED"
    )
    audit_ok = (
        len(endpoints) == 1
        and len(initialization) == 1
        and len(frozen_initialization) == 1
        and None not in frozen_initialization
        and len(heldout_indices) == 1
        and "null" not in heldout_indices
        and heldout_contract_ok
        and heldout_arrays_ok
        and arm_identity_ok
        and frozen_unchanged
        and all_arms_complete
        and baseline_platformed
    )
    endpoint = endpoints.pop() if len(endpoints) == 1 else None

    reliance = paired_bootstrap_reliance_change(
        evidence["A"]["final"]["losses"],
        evidence["B30"]["final"]["losses"],
        draws=bootstrap_draws,
        seed=seed,
    )
    wan_learning = _paired_mean_change(
        evidence["B30"]["step_zero"]["losses"]["wan_drop"],
        evidence["B30"]["final"]["losses"]["wan_drop"],
        draws=bootstrap_draws,
        seed=seed + 1,
    )
    kill = random_patch_kill_test(
        baseline=evidence["A"]["final"]["losses"],
        semantic=evidence["B30"]["final"]["losses"],
        random_patch=evidence["D"]["final"]["losses"],
        draws=bootstrap_draws,
        seed=seed + 2,
    )
    rollout_summary = _rollout_summary(rollout)
    rollout_arms = rollout_summary["arms"]
    clean_sr_drop_pp = None
    if "A" in rollout_arms and "B30" in rollout_arms:
        clean_sr_drop_pp = 100.0 * (
            rollout_arms["A"]["clean"]["success_rate"]
            - rollout_arms["B30"]["clean"]["success_rate"]
        )
    language_summary = _language_summary(language_canary)
    language_complete = (
        isinstance(language_summary, Mapping)
        and language_summary.get("status") == "COMPLETE"
    )
    language_degraded = (
        bool(language_summary.get("material_degradation"))
        if language_complete
        else None
    )

    d_a = float(evidence["A"]["final"]["d_dino_loss"])
    d_b = float(evidence["B30"]["final"]["d_dino_loss"])
    reliance_go = d_b >= 1.25 * d_a and reliance["ci95"][0] > 0.0
    wan_go = wan_learning["ci95"][1] < 0.0
    clean_go = clean_sr_drop_pp is not None and clean_sr_drop_pp <= 5.0
    language_go = language_complete and language_degraded is False
    kill_go = kill["outcome"] == "NOT_REPRODUCED"

    no_go_reasons = []
    if kill["outcome"] == "REPRODUCED":
        no_go_reasons.append("random patches reproduced at least 80% of reliance gain")
    if reliance["ci95"][1] <= 0.0:
        no_go_reasons.append("B30 DINO reliance did not increase")
    if clean_sr_drop_pp is not None and clean_sr_drop_pp > 5.0:
        no_go_reasons.append("B30 clean SR drop exceeded 5 percentage points")
    clean_loss_by_probability = [
        float(evidence[arm]["final"]["loss"]["clean"])
        for arm in ("B15", "B30", "B50")
    ]
    clean_sr_by_probability = None
    clean_sr_monotonic_over_tolerance = False
    if all(arm in rollout_arms for arm in ("A", "B15", "B30", "B50")):
        clean_sr_by_probability = [
            rollout_arms[arm]["clean"]["success_rate"]
            for arm in ("A", "B15", "B30", "B50")
        ]
        monotonic = all(
            left >= right
            for left, right in zip(
                clean_sr_by_probability[:-1],
                clean_sr_by_probability[1:],
                strict=True,
            )
        )
        clean_sr_monotonic_over_tolerance = (
            monotonic
            and 100.0
            * (clean_sr_by_probability[0] - clean_sr_by_probability[-1])
            > 5.0
        )
    if clean_sr_monotonic_over_tolerance:
        no_go_reasons.append(
            "clean SR worsened monotonically with p and exceeded 5 percentage points"
        )
    if language_degraded is True:
        no_go_reasons.append("language canary materially degraded")
    if no_go_reasons:
        decision = "NO-GO"
    elif audit_ok and reliance_go and wan_go and clean_go and language_go and kill_go:
        decision = "GO"
    else:
        decision = "EVIDENCE_INSUFFICIENT"

    table = []
    for arm in PILOT_ARMS:
        zero = evidence[arm]["step_zero"]
        final = evidence[arm]["final"]
        diagnostics = final["diagnostics"]["overall"]
        wan_change = _paired_mean_change(
            zero["losses"]["wan_drop"],
            final["losses"]["wan_drop"],
            draws=bootstrap_draws,
            seed=seed + 100 + len(table),
        )
        arm_rollout = rollout_arms.get(arm)
        arm_clean_drop = None
        if arm_rollout is not None and "A" in rollout_arms:
            arm_clean_drop = 100.0 * (
                rollout_arms["A"]["clean"]["success_rate"]
                - arm_rollout["clean"]["success_rate"]
            )
        ood_direction = "NOT-RUN"
        if ood is not None:
            ood_direction = ood.get("arms", {}).get(arm, "NOT-RUN")
        table.append(
            {
                "arm": arm,
                "K": int(evidence[arm]["result"]["global_step"]),
                "clean_loss": float(final["loss"]["clean"]),
                "wan_drop_loss_change": wan_change,
                "d_dino": float(final["d_dino_loss"]),
                "d_wan": float(final["d_wan_loss"]),
                "clean_sr_delta_pp": arm_clean_drop,
                "gate_mean": float(diagnostics["gate_mean"]),
                "projected_norm": float(diagnostics["projected_norm"]),
                "residual_norm": float(diagnostics["residual_norm"]),
                "residual_cross_sample_variance": float(
                    diagnostics["residual_cross_sample_variance"]
                ),
                "ood_direction": ood_direction,
                "mechanism_diagnosis": _mechanism_diagnosis(zero, final),
                "decision": decision if arm == "B30" else "DESCRIPTIVE",
            }
        )
    return {
        "schema": "fastwam-modality-dropout-pilot-decision-v1",
        "decision": decision,
        "no_go_reasons": no_go_reasons,
        "audit": {
            "passed": audit_ok,
            "common_endpoint": endpoint,
            "common_initialization": len(initialization) == 1,
            "common_frozen_initialization": len(frozen_initialization) == 1,
            "common_heldout_indices": len(heldout_indices) == 1,
            "heldout_contract": heldout_contract_ok,
            "heldout_arrays": heldout_arrays_ok,
            "arm_identity": arm_identity_ok,
            "frozen_unchanged": frozen_unchanged,
            "all_arms_complete": all_arms_complete,
            "baseline_platformed": baseline_platformed,
        },
        "b30_vs_a_d_dino": reliance,
        "b30_wan_drop_loss_learning": wan_learning,
        "random_patch_kill": kill,
        "clean_sr_drop_pp": clean_sr_drop_pp,
        "clean_loss_by_probability": clean_loss_by_probability,
        "clean_sr_by_probability": clean_sr_by_probability,
        "clean_sr_monotonic_over_tolerance": clean_sr_monotonic_over_tolerance,
        "language_canary": language_summary,
        "rollout": rollout_summary,
        "ood": ood or "NOT-RUN",
        "table": table,
    }
