#!/usr/bin/env bash
# SETTING: Two independent S-DR pilot trainings with the preregistered 1e-5 and 3e-5 base learning rates.
# MODEL/CHECKPOINT LINEAGE: The same standalone E-I parent initializes both pilots; neither pilot resumes from the other.
# SCIENTIFIC GOAL: Compare optimization stability before committing endpoint/control/Gate artifacts to one shared checkpoint.
# ACCEPTANCE: Both pilots finish the full optimizer-step schedule with finite weights; selection binds one exact checkpoint SHA.
# REQUIRED INPUTS: All run_e1_train_shared.sh inputs plus SELECTED_LR (1e-5 or 3e-5) and a non-empty SELECTION_BASIS.
# OUTPUTS: Two independent S-DR runs and pilot_selection.json consumed by E1 endpoint/profile launchers.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
for name in E_I_CKPT E_I_CONFIG DATASET_STATS WARMSTART_DECISION SELECTED_LR SELECTION_BASIS; do
    require_env "${name}"
done
[[ "${SELECTED_LR}" == "1e-5" || "${SELECTED_LR}" == "3e-5" ]] || \
    die "SELECTED_LR must be 1e-5 or 3e-5"
[[ -n "${SELECTION_BASIS}" ]] || die "SELECTION_BASIS must explain the pilot choice"

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${EXPERIMENT_ROOT}/E1_shared_pilot_selection/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"
CANDIDATE_1E5_ID="${RUN_ID}_lr1e-5"
CANDIDATE_3E5_ID="${RUN_ID}_lr3e-5"
CANDIDATE_1E5_DIR="${EXPERIMENT_ROOT}/E1_shared_train/${CANDIDATE_1E5_ID}"
CANDIDATE_3E5_DIR="${EXPERIMENT_ROOT}/E1_shared_train/${CANDIDATE_3E5_ID}"

build_candidate_command() {
    local lr=$1
    local child_id=$2
    local devices=$3
    CANDIDATE_CMD=(
        env "BASE_LR=${lr}" "RUN_ID=${child_id}"
    )
    if [[ -n "${devices}" ]]; then
        CANDIDATE_CMD+=("CUDA_VISIBLE_DEVICES=${devices}")
    fi
    CANDIDATE_CMD+=(bash scripts/adaptive_gate/run_e1_train_shared.sh)
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        CANDIDATE_CMD+=(--dry-run)
    fi
    CANDIDATE_CMD+=(-- "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}")
}

build_candidate_command 1e-5 "${CANDIDATE_1E5_ID}" "${PILOT_1E5_VISIBLE_DEVICES:-}"
CMD_1E5=("${CANDIDATE_CMD[@]}")
run_command "${CMD_1E5[@]}"
build_candidate_command 3e-5 "${CANDIDATE_3E5_ID}" "${PILOT_3E5_VISIBLE_DEVICES:-}"
CMD_3E5=("${CANDIDATE_CMD[@]}")
run_command "${CMD_3E5[@]}"

VALIDATION_1E5="${CANDIDATE_1E5_DIR}/completion.json"
VALIDATION_3E5="${CANDIDATE_3E5_DIR}/completion.json"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    require_file "${VALIDATION_1E5}"
    require_file "${VALIDATION_3E5}"
fi
SELECT_CMD=(
    python scripts/validate_sdr_checkpoint.py select
    --candidate-validation "${VALIDATION_1E5}"
    --candidate-validation "${VALIDATION_3E5}"
    --selected-lr "${SELECTED_LR}"
    --selection-basis "${SELECTION_BASIS}"
    --out "${RUN_DIR}/pilot_selection.json"
)
run_command "${SELECT_CMD[@]}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    RUN_ARTIFACTS=(
        "${VALIDATION_1E5}" "${VALIDATION_3E5}"
        "${CANDIDATE_1E5_DIR}/run_manifest.json"
        "${CANDIDATE_3E5_DIR}/run_manifest.json"
        "${RUN_DIR}/pilot_selection.json"
    )
fi
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
