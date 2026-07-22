#!/usr/bin/env python3
"""Apply the preregistered E1-P2 final-checkpoint acceptance contract."""
from __future__ import annotations

import argparse
import json
import torch

from decide_sdr_learning_probe import _diagnostic_summary, _training_summary
from fastwam.adaptive_gate.sdr_contracts import (
    FORMAL_TRAINING_DECISION_SCHEMA,
    artifact_record,
    atomic_json,
    read_json,
    validate_formal_training_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-decision", required=True)
    parser.add_argument("--learning-probe-decision", required=True)
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--baseline-diagnostics", required=True)
    parser.add_argument("--final-diagnostics", required=True)
    parser.add_argument("--training-metrics", required=True)
    parser.add_argument("--final-delta")
    parser.add_argument("--final-checkpoint", required=True)
    parser.add_argument("--reconstruction-decision", required=True)
    parser.add_argument("--checkpoint-completion", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    preflight = read_json(args.preflight_decision)
    probe = read_json(args.learning_probe_decision)
    lineage = read_json(args.lineage_manifest)
    contract = validate_formal_training_contract(
        preflight_decision=preflight,
        learning_probe_decision=probe,
        lineage_manifest=lineage,
    )
    completion = read_json(args.checkpoint_completion)
    if completion.get("status") != "PASS":
        raise ValueError("Formal final checkpoint completion is not PASS.")
    final_payload = torch.load(
        args.final_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    provenance = final_payload.get("fastwam_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Formal final checkpoint has no FastWAM provenance.")
    expected_steps = int(
        provenance["dual_regime_training_contract"]["total_optimizer_steps"]
    )
    if int(final_payload.get("step")) != expected_steps:
        raise ValueError(
            "Formal final checkpoint did not complete all optimizer steps."
        )
    training = _training_summary(
        args.training_metrics,
        expected_steps=expected_steps,
        schedule=contract["schedule"],
        schedule_total_steps=expected_steps,
    )
    baseline = _diagnostic_summary(
        args.baseline_diagnostics,
        weight=float(preflight["w_cap"]),
    )
    final = _diagnostic_summary(
        args.final_diagnostics,
        weight=float(preflight["w_cap"]),
    )
    failures = []
    if final["uncond_raw_loss"] >= baseline["uncond_raw_loss"]:
        failures.append("UNCOND held-out raw loss did not improve")
    if final["idm_raw_loss"] > 1.05 * baseline["idm_raw_loss"]:
        failures.append("GT-IDM raw loss worsened by more than 5%")
    if final["generated_idm_action_l2"] > 1.05 * baseline[
        "generated_idm_action_l2"
    ]:
        failures.append("generated-IDM action error worsened by more than 5%")
    if final["sensitivity_median"] < 0.5 * baseline["sensitivity_median"]:
        failures.append("valid/no-read sensitivity retained less than 50%")
    if final["common_action_all_idm_margin"] <= 0.0:
        failures.append("final common-noise IDM margin is not positive")
    if final["final_blocks_negative_idm_margin_fraction"] >= 0.20:
        failures.append("final-block negative IDM margin fraction reached 20%")
    if not final["no_read_uncond_parity"]:
        failures.append("no-read/forced-UNCOND parity failed")
    if not training["all_finite"] or not final["finite"]:
        failures.append("non-finite metric")
    if completion.get("training_contract_kind") != "single_formal_run":
        failures.append("checkpoint used the wrong training contract")
    final_checkpoint = artifact_record(args.final_checkpoint)
    if completion.get("checkpoint_sha256") != final_checkpoint["sha256"]:
        raise ValueError("Formal completion binds a different final checkpoint.")
    reconstruction = read_json(args.reconstruction_decision)
    if (
        reconstruction.get("status") != "PASS"
        or reconstruction.get("output_checkpoint_sha256")
        != final_checkpoint["sha256"]
        or reconstruction.get("parent_checkpoint_sha256")
        != contract["artifact_bindings"]["e_i_checkpoint"]
    ):
        raise ValueError("Formal final reconstruction evidence is inconsistent.")
    result = {
        "schema": FORMAL_TRAINING_DECISION_SCHEMA,
        "status": "PASS" if not failures else "FAIL-DIAGNOSED",
        "plus_full_used": False,
        "artifact_bindings": preflight["artifact_bindings"],
        "initializer_stage": "e_i_s0",
        "formal_training": "PASS" if not failures else "FAIL-DIAGNOSED",
        "completed_optimizer_steps": expected_steps,
        "num_epochs": 10,
        "base_learning_rate": 1e-5,
        "training": training,
        "baseline": baseline,
        "final": final,
        "failure_conditions": failures,
        "final_checkpoint": final_checkpoint,
        "final_checkpoint_kind": "full_weights",
        "final_delta": (
            artifact_record(args.final_delta) if args.final_delta else None
        ),
        "reconstruction_decision": artifact_record(
            args.reconstruction_decision
        ),
        "checkpoint_completion": artifact_record(args.checkpoint_completion),
        "resolved_config": artifact_record(args.resolved_config),
        "no_plus_full_statement": (
            "No Plus-Full or task-test outcome selected this checkpoint."
        ),
    }
    atomic_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_optimizer_steps": expected_steps,
                "final_checkpoint_sha256": final_checkpoint["sha256"],
            },
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
