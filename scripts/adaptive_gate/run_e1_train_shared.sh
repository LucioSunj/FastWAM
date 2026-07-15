#!/usr/bin/env bash
# SETTING: Standard LIBERO WAM-Train, fused shared UNCOND/IDM model, staged optimizer-step schedule.
# MODEL/CHECKPOINT LINEAGE: standalone E-I -> strict S0 import -> S-DR; E-U is not an initializer.
# SCIENTIFIC GOAL: Preserve future-reading while gradually introducing the cheap UNCOND regime.
# ACCEPTANCE: Strict provenance, positive dual_regime_optimizer_steps, finite losses, full schedule completion.
# REQUIRED INPUTS: E_I_CKPT, E_I_CONFIG, DATASET_STATS, WARMSTART_DECISION; optional BASE_LR/SEED/NPROC_PER_NODE.
# OUTPUTS: S-DR weights/state checkpoints, diagnostics, resolved config, completion validation, run_manifest.json.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_env E_I_CKPT
require_env E_I_CONFIG
require_env DATASET_STATS
require_env WARMSTART_DECISION
require_file "${E_I_CKPT}"
require_file "${E_I_CONFIG}"
require_file "${DATASET_STATS}"
require_passed_decision "${WARMSTART_DECISION}"

TASK=${TASK:-libero_dual_regime_fused_2cam224_1e-4}
SOURCE_TASK=${SOURCE_TASK:-libero_idm_2cam224_1e-4}
BASE_LR=${BASE_LR:-1e-5}
SEED=${SEED:-42}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)_lr${BASE_LR}_seed${SEED}}
RUN_DIR="${EXPERIMENT_ROOT}/E1_shared_train/${RUN_ID}"
prepare_run_dir "${RUN_DIR}"
E_I_SHA256=${E_I_SHA256:-$(python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${E_I_CKPT}")}
RUN_ARTIFACTS=(
    "${E_I_CKPT}" "${E_I_CONFIG}" "${DATASET_STATS}" "${WARMSTART_DECISION}"
)

CONFIG_OVERRIDES=(
    "task=${TASK}"
    "output_dir=${RUN_DIR}"
    "learning_rate=${BASE_LR}"
    "seed=${SEED}"
    "data.train.pretrained_norm_stats=${DATASET_STATS}"
    "warm_start.kind=standalone_idm"
    "warm_start.checkpoint=${E_I_CKPT}"
    "warm_start.expected_checkpoint_sha256=${E_I_SHA256}"
    "warm_start.source_task=${SOURCE_TASK}"
    "warm_start.source_config=${E_I_CONFIG}"
    "warm_start.source_dataset_stats=${DATASET_STATS}"
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
)
resolve_hydra_config "${PROJECT_REPO_ROOT}/configs" train \
    "${RUN_DIR}/resolved_config.yaml" "${CONFIG_OVERRIDES[@]}"

CMD=(
    env "RUN_ID=${RUN_ID}"
    bash scripts/train_zero1.sh "${NPROC_PER_NODE}"
    "${CONFIG_OVERRIDES[@]}"
)
run_command "${CMD[@]}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    FINAL_CHECKPOINT=${FINAL_CHECKPOINT:-$(python -c 'import glob,sys; p=sorted(glob.glob(sys.argv[1])); print(p[-1] if p else "")' "${RUN_DIR}/checkpoints/weights/step_*.pt")}
    [[ -n "${FINAL_CHECKPOINT}" ]] || die "S-DR training produced no weights checkpoint"
    require_file "${FINAL_CHECKPOINT}"
else
    FINAL_CHECKPOINT=${FINAL_CHECKPOINT:-"${RUN_DIR}/checkpoints/weights/<final-step>.pt"}
fi
VALIDATE_CMD=(
    python scripts/validate_sdr_checkpoint.py validate
    --checkpoint "${FINAL_CHECKPOINT}"
    --resolved-config "${RUN_DIR}/resolved_config.yaml"
    --dataset-stats "${DATASET_STATS}"
    --expected-base-lr "${BASE_LR}"
    --out "${RUN_DIR}/completion.json"
)
run_command "${VALIDATE_CMD[@]}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    RUN_ARTIFACTS+=("${FINAL_CHECKPOINT}" "${RUN_DIR}/completion.json")
fi
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
