#!/usr/bin/env bash
# SETTING: Frozen official Plus-Full episodes, paired reset/seed order, synchronized end-to-end chunk timing.
# MODEL/CHECKPOINT LINEAGE: separately trained E-U and E-I only; the higher-quality E-U remains the conservative comparator.
# SCIENTIFIC GOAL: Lock the motivating endpoint success/compute trade-off before shared training.
# ACCEPTANCE: Paired Delta_ref has positive CI and IDM mean chunk latency exceeds UNCOND; otherwise E0 is AMBER/FAIL.
# REQUIRED INPUTS: E_U_CKPT/E_U_STATS, E_I_CKPT/E_I_STATS, PLUS_FULL_MANIFEST, LIBERO_PLUS_ROOT/COMMIT.
# OUTPUTS: per-endpoint shard JSONL/summary, merged endpoint_trials.jsonl, decision.json, resolved configs and run manifests.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
for name in E_U_CKPT E_U_STATS E_I_CKPT E_I_STATS PLUS_FULL_MANIFEST; do
    require_env "${name}"
    require_file "${!name}"
done
validate_plus_manifest "${PLUS_FULL_MANIFEST}"

E_U_TASK=${E_U_TASK:-libero_uncond_2cam224_1e-4}
E_I_TASK=${E_I_TASK:-libero_idm_2cam224_1e-4}
NUM_SHARDS=${NUM_SHARDS:-1}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${EXPERIMENT_ROOT}/E0_standalone_endpoints/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"

U_DIR="${RUN_DIR}/standalone_uncond"
prepare_run_dir "${U_DIR}"
U_SCOPE_START=${#RUN_COMMAND_LOG[@]}
RUN_ARTIFACTS=("${E_U_CKPT}" "${E_U_STATS}" "${PLUS_FULL_MANIFEST}")
resolve_hydra_config "${PROJECT_REPO_ROOT}/configs" sim_libero \
    "${U_DIR}/resolved_config.yaml" "task=${E_U_TASK}" \
    "EVALUATION.dataset_stats_path=${E_U_STATS}" \
    "EVALUATION.force_branch=base" \
    "EVALUATION.num_inference_steps=${INFERENCE_STEPS:-20}" \
    "EVALUATION.sigma_shift=${SIGMA_SHIFT:-null}" \
    "EVALUATION.replan_steps=${REPLAN_STEPS:-10}"
run_plus_endpoint_shards standalone_uncond "${E_U_TASK}" "${E_U_CKPT}" \
    "${E_U_STATS}" base "${PLUS_FULL_MANIFEST}" "${U_DIR}" \
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
merge_endpoint_shards "${U_DIR}" standalone_uncond "${U_DIR}/trials.jsonl"
write_scoped_run_manifest "${U_DIR}/run_manifest.json" "${U_SCOPE_START}"

I_DIR="${RUN_DIR}/standalone_idm"
prepare_run_dir "${I_DIR}"
I_SCOPE_START=${#RUN_COMMAND_LOG[@]}
RUN_ARTIFACTS=("${E_I_CKPT}" "${E_I_STATS}" "${PLUS_FULL_MANIFEST}")
resolve_hydra_config "${PROJECT_REPO_ROOT}/configs" sim_libero \
    "${I_DIR}/resolved_config.yaml" "task=${E_I_TASK}" \
    "EVALUATION.dataset_stats_path=${E_I_STATS}" \
    "EVALUATION.force_branch=idm" \
    "EVALUATION.num_inference_steps=${INFERENCE_STEPS:-20}" \
    "EVALUATION.sigma_shift=${SIGMA_SHIFT:-null}" \
    "EVALUATION.replan_steps=${REPLAN_STEPS:-10}"
run_plus_endpoint_shards standalone_idm "${E_I_TASK}" "${E_I_CKPT}" \
    "${E_I_STATS}" idm "${PLUS_FULL_MANIFEST}" "${I_DIR}" \
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
merge_endpoint_shards "${I_DIR}" standalone_idm "${I_DIR}/trials.jsonl"
write_scoped_run_manifest "${I_DIR}/run_manifest.json" "${I_SCOPE_START}"

MERGE_CMD=(
    python "${MERGE_JSONL_TOOL}"
    --input "${U_DIR}/trials.jsonl"
    --input "${I_DIR}/trials.jsonl"
    --out "${RUN_DIR}/endpoint_trials.jsonl"
)
run_command "${MERGE_CMD[@]}"
DECISION_CMD=(
    python "${DECISION_TOOL}" e0
    --trials "${RUN_DIR}/endpoint_trials.jsonl"
    --out "${RUN_DIR}/decision.json"
)
run_command "${DECISION_CMD[@]}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    RUN_ARTIFACTS=(
        "${E_U_CKPT}" "${E_U_STATS}" "${E_I_CKPT}" "${E_I_STATS}"
        "${PLUS_FULL_MANIFEST}" "${U_DIR}/trials.jsonl" "${I_DIR}/trials.jsonl"
        "${RUN_DIR}/endpoint_trials.jsonl" "${RUN_DIR}/decision.json"
    )
fi
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
