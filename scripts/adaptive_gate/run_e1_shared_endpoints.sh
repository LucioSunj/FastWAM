#!/usr/bin/env bash
# SETTING: Frozen official Plus-Full episodes, paired with the exact E0 endpoint trial identities.
# MODEL/CHECKPOINT LINEAGE: final trained S-DR forced base/IDM branches; S0 and standalone checkpoints are rejected as shared endpoints.
# SCIENTIFIC GOAL: Verify the deployable shared model retains the endpoint treatment gap needed for adaptive allocation.
# ACCEPTANCE: CI_low(Delta_shared)>0, retention point>=0.8/lower bound>=0.5, and both branches meet 5pp non-inferiority.
# REQUIRED INPUTS: SHARED_CKPT, SHARED_STATS, S_DR_SELECTION, E0_TRIALS, E0_DECISION, PLUS_FULL_MANIFEST and pinned LIBERO-Plus checkout.
# OUTPUTS: paired shared endpoint JSONL, combined 2x2 trials, E1 decision.json, resolved config and run manifests.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
for name in SHARED_CKPT SHARED_STATS E0_TRIALS E0_DECISION PLUS_FULL_MANIFEST; do
    require_env "${name}"
    require_file "${!name}"
done
require_passed_decision "${E0_DECISION}"
validate_plus_manifest "${PLUS_FULL_MANIFEST}"
require_selected_sdr_checkpoint "${SHARED_CKPT}"

TASK=${TASK:-libero_dual_regime_fused_2cam224_1e-4}
NUM_SHARDS=${NUM_SHARDS:-1}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${EXPERIMENT_ROOT}/E1_shared_endpoints/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"

for spec in "shared_uncond:base" "shared_idm:idm"; do
    method=${spec%%:*}
    branch=${spec##*:}
    METHOD_DIR="${RUN_DIR}/${method}"
    prepare_run_dir "${METHOD_DIR}"
    METHOD_SCOPE_START=${#RUN_COMMAND_LOG[@]}
    resolve_hydra_config "${PROJECT_REPO_ROOT}/configs" sim_libero \
        "${METHOD_DIR}/resolved_config.yaml" "task=${TASK}" \
        "EVALUATION.dataset_stats_path=${SHARED_STATS}" \
        "EVALUATION.force_branch=${branch}" \
        "EVALUATION.num_inference_steps=${INFERENCE_STEPS:-20}" \
        "EVALUATION.sigma_shift=${SIGMA_SHIFT:-null}" \
        "EVALUATION.replan_steps=${REPLAN_STEPS:-10}"
    RUN_ARTIFACTS=(
        "${SHARED_CKPT}" "${SHARED_STATS}" "${PLUS_FULL_MANIFEST}"
        "${E0_TRIALS}" "${E0_DECISION}"
        "${S_DR_SELECTION}"
    )
    run_plus_endpoint_shards "${method}" "${TASK}" "${SHARED_CKPT}" \
        "${SHARED_STATS}" "${branch}" "${PLUS_FULL_MANIFEST}" "${METHOD_DIR}" \
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
    merge_endpoint_shards "${METHOD_DIR}" "${method}" "${METHOD_DIR}/trials.jsonl"
    write_scoped_run_manifest \
        "${METHOD_DIR}/run_manifest.json" "${METHOD_SCOPE_START}"
done

MERGE_CMD=(
    python "${MERGE_JSONL_TOOL}"
    --input "${E0_TRIALS}"
    --input "${RUN_DIR}/shared_uncond/trials.jsonl"
    --input "${RUN_DIR}/shared_idm/trials.jsonl"
    --out "${RUN_DIR}/endpoint_2x2_trials.jsonl"
)
run_command "${MERGE_CMD[@]}"
DECISION_CMD=(
    python "${DECISION_TOOL}" e1
    --trials "${RUN_DIR}/endpoint_2x2_trials.jsonl"
    --out "${RUN_DIR}/decision.json"
    --retention-point 0.8 --retention-low 0.5 --noninferiority-margin 0.05
)
run_command "${DECISION_CMD[@]}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    RUN_ARTIFACTS=(
        "${SHARED_CKPT}" "${SHARED_STATS}" "${S_DR_SELECTION}" "${PLUS_FULL_MANIFEST}"
        "${E0_TRIALS}" "${RUN_DIR}/shared_uncond/trials.jsonl"
        "${RUN_DIR}/shared_idm/trials.jsonl" "${RUN_DIR}/endpoint_2x2_trials.jsonl"
        "${RUN_DIR}/decision.json"
    )
fi
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
