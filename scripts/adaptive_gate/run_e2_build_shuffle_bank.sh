#!/usr/bin/env bash
# SETTING: Offline packing of reliable-phase future latents captured only by the valid_idm E2 evaluation path.
# MODEL/CHECKPOINT LINEAGE: final trained S-DR + exact dataset stats/control profile -> provenance-bound shuffled-future bank.
# SCIENTIFIC GOAL: Build the content-intervention donor bank without mixing checkpoints, solver seeds, cells, or post-choice states.
# ACCEPTANCE: E1 is PASS; every donor is schema-v1 valid_idm capture; lineage matches; every task/factor/level/phase cell has >=2 states.
# REQUIRED INPUTS: DONOR_GLOB, CONTROL_PROFILE, SHARED_CKPT, S_DR_SELECTION, DATASET_STATS, and a PASS E1_DECISION.
# OUTPUTS: shuffled_future_bank.pt, resolved_cli_config.json, run_manifest.json, and decision.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"

for name in DONOR_GLOB CONTROL_PROFILE SHARED_CKPT DATASET_STATS E1_DECISION; do
    require_env "${name}"
done
require_glob "${DONOR_GLOB}"
for path in "${CONTROL_PROFILE}" "${SHARED_CKPT}" "${DATASET_STATS}"; do
    require_file "${path}"
done
require_passed_decision "${E1_DECISION}"
require_selected_sdr_checkpoint "${SHARED_CKPT}"

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${EXPERIMENT_ROOT}/E2_shuffle_bank/${RUN_ID}"
BANK_PATH="${RUN_DIR}/shuffled_future_bank.pt"
prepare_run_dir "${RUN_DIR}"

freeze_cli_config "${RUN_DIR}/resolved_cli_config.json" \
    "stage=E2_shuffle_bank" \
    "donor_source=valid_idm_capture" \
    "donor_glob=${DONOR_GLOB}" \
    "control_profile=${CONTROL_PROFILE}" \
    "shared_ckpt=${SHARED_CKPT}" \
    "dataset_stats=${DATASET_STATS}" \
    "e1_decision=${E1_DECISION}" \
    "minimum_states_per_task_factor_level_phase_cell=2" \
    "output_bank=${BANK_PATH}"

RUN_ARTIFACTS=(
    "${CONTROL_PROFILE}"
    "${SHARED_CKPT}"
    "${DATASET_STATS}"
    "${E1_DECISION}"
    "${S_DR_SELECTION}"
)
add_glob_artifacts "${DONOR_GLOB}"

CMD=(
    python scripts/build_shuffled_future_bank.py
    --inputs "${DONOR_GLOB}"
    --profile "${CONTROL_PROFILE}"
    --shared-ckpt "${SHARED_CKPT}"
    --dataset-stats "${DATASET_STATS}"
    --out "${BANK_PATH}"
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
)
run_command "${CMD[@]}"
RUN_ARTIFACTS+=("${BANK_PATH}")
run_command python "${DECISION_TOOL}" contract \
    --check e2_valid_idm_shuffled_future_bank \
    --evidence "${BANK_PATH}" \
    --evidence "${CONTROL_PROFILE}" \
    --out "${RUN_DIR}/decision.json"
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
