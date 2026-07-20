#!/usr/bin/env bash
# SETTING: Real E-I dataset state, fixed WAM/action seed, identical complete IDM solver.
# MODEL/CHECKPOINT LINEAGE: standalone E-I -> strict weight-only S0 import; no optimizer/scheduler/step inheritance.
# SCIENTIFIC GOAL: Prove S0 forced-IDM is numerically the same model as its E-I parent before S-DR training.
# ACCEPTANCE: Strict provenance/import checks pass and action tensors satisfy configured atol/rtol.
# REQUIRED INPUTS: E_I_CKPT, E_I_CONFIG, DATASET_STATS, P0_FUSED_DECISION and real Wan2.2/ActionDiT/data assets.
# OUTPUTS: parity result, fully resolved target config, decision.json and run_manifest.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_env E_I_CKPT
require_env E_I_CONFIG
require_env DATASET_STATS
require_file "${E_I_CKPT}"
require_file "${E_I_CONFIG}"
require_file "${DATASET_STATS}"
require_env P0_FUSED_DECISION
require_passed_decision "${P0_FUSED_DECISION}"

SOURCE_TASK=${SOURCE_TASK:-libero_idm_2cam224_1e-4}
TARGET_TASK=${TARGET_TASK:-libero_dual_regime_fused_2cam224_1e-4}
SAMPLE_INDEX=${SAMPLE_INDEX:-0}
INFERENCE_STEPS=${INFERENCE_STEPS:-20}
SEED=${SEED:-0}
DTYPE=${DTYPE:-bfloat16}
ATOL=${ATOL:-5e-4}
RTOL=${RTOL:-5e-3}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="${EXPERIMENT_ROOT}/E1_warmstart_parity/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"
PARITY_RESULT="${RUN_DIR}/parity_result.json"
E_I_SHA256=${E_I_SHA256:-$(python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${E_I_CKPT}")}
RUN_ARTIFACTS=(
    "${E_I_CKPT}" "${E_I_CONFIG}" "${DATASET_STATS}" "${P0_FUSED_DECISION}"
)
resolve_hydra_config "${PROJECT_REPO_ROOT}/configs" train \
    "${RUN_DIR}/resolved_config.yaml" "task=${TARGET_TASK}"

CMD=(
    python scripts/verify_dual_regime_warm_start.py
    --source-config "${E_I_CONFIG}"
    --source-task "${SOURCE_TASK}"
    --target-task "${TARGET_TASK}"
    --ckpt "${E_I_CKPT}"
    --checkpoint-sha256 "${E_I_SHA256}"
    --dataset-stats "${DATASET_STATS}"
    --sample-index "${SAMPLE_INDEX}"
    --inference-steps "${INFERENCE_STEPS}"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --atol "${ATOL}"
    --rtol "${RTOL}"
    --out "${PARITY_RESULT}"
)
if [[ -n "${SIGMA_SHIFT:-}" && "${SIGMA_SHIFT}" != "null" ]]; then
    CMD+=(--sigma-shift "${SIGMA_SHIFT}")
fi
CMD+=("${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}")
run_command "${CMD[@]}"
RUN_ARTIFACTS+=("${PARITY_RESULT}")
run_command python "${DECISION_TOOL}" contract \
    --check standalone_idm_to_s0_fixed_seed_parity \
    --evidence "${PARITY_RESULT}" \
    --out "${RUN_DIR}/decision.json"
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
