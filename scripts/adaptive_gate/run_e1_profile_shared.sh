#!/usr/bin/env bash
# SETTING: Synthetic fixed-shape profiling on the target deployment GPU; not headline closed-loop timing.
# MODEL/CHECKPOINT LINEAGE: Final frozen S-DR checkpoint only.
# SCIENTIFIC GOAL: Bind production mode costs and calibrate NoRead/ExtraCompute controls.
# ACCEPTANCE: Production profile is valid; NoRead and ExtraCompute latency are within 5% of IDM.
# REQUIRED INPUTS: SHARED_CKPT, DATASET_STATS, S_DR_SELECTION; target GPU and matching dtype/solver/shape.
# OUTPUTS: wam_cost.yaml, wam_controls.yaml, run_manifest.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_env SHARED_CKPT
require_env DATASET_STATS
require_file "${SHARED_CKPT}"
require_file "${DATASET_STATS}"
require_selected_sdr_checkpoint "${SHARED_CKPT}"

TASK=${TASK:-libero_dual_regime_fused_2cam224_1e-4}
INFERENCE_STEPS=${INFERENCE_STEPS:-20}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${EXPERIMENT_ROOT}/E1_shared_profile/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"
STATS_SHA256=$(python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${DATASET_STATS}")
RUN_ARTIFACTS=("${SHARED_CKPT}" "${DATASET_STATS}" "${S_DR_SELECTION}")
resolve_hydra_config "${PROJECT_REPO_ROOT}/configs" train \
    "${RUN_DIR}/resolved_config.yaml" "task=${TASK}" \
    "eval_num_inference_steps=${INFERENCE_STEPS}" \
    "eval_sigma_shift=${SIGMA_SHIFT:-null}"

MODE_CMD=(
    python scripts/profile_wam_modes.py
    --task "${TASK}" --backbone-kind idm --ckpt "${SHARED_CKPT}"
    --inference-steps "${INFERENCE_STEPS}" --height 224 --width 448
    --num-video-frames 9 --action-horizon 32 --out "${RUN_DIR}/wam_cost.yaml"
)
CONTROL_CMD=(
    python scripts/profile_wam_controls.py
    --task "${TASK}" --ckpt "${SHARED_CKPT}"
    --dataset-stats-sha256 "${STATS_SHA256}"
    --inference-steps "${INFERENCE_STEPS}" --height 224 --width 448
    --num-video-frames 9 --action-horizon 32 --latency-match-tolerance 0.05
    --out "${RUN_DIR}/wam_controls.yaml"
)
if [[ -n "${SIGMA_SHIFT:-}" && "${SIGMA_SHIFT}" != "null" ]]; then
    MODE_CMD+=(--sigma-shift "${SIGMA_SHIFT}")
    CONTROL_CMD+=(--sigma-shift "${SIGMA_SHIFT}")
fi
CONTROL_CMD+=("${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}")
run_command "${MODE_CMD[@]}"
run_command "${CONTROL_CMD[@]}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    RUN_ARTIFACTS+=("${RUN_DIR}/wam_cost.yaml" "${RUN_DIR}/wam_controls.yaml")
fi
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
