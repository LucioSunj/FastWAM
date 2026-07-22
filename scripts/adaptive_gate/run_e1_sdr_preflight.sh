#!/usr/bin/env bash
# E1-P0.5: strict lineage gate, fixed validation, no-update gradients and controls.
set -euo pipefail
: "${EXPERIMENT_ROOT:=/root/autodl-tmp/experiments/adaptive_wm_reasoning}"
export EXPERIMENT_ROOT
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"

for name in \
    E_I_BASE_MODEL_MANIFEST E_I_CKPT E_I_CONFIG \
    E_I_LINEAGE_MANIFEST DATASET_STATS WARMSTART_DECISION SDR_VAL_MANIFEST; do
    require_env "${name}"
    require_file "${!name}"
done

TASK=${TASK:-libero_dual_regime_fused_2cam224_1e-4}
SOURCE_TASK=${SOURCE_TASK:-libero_idm_2cam224_1e-4}
SEED=${SEED:-20260721}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="${EXPERIMENT_ROOT}/E1_P0_5/${RUN_ID}"
LINEAGE_AUDIT="${RUN_DIR}/e_i_lineage_audit.json"
prepare_run_dir "${RUN_DIR}"

AUDIT_CMD=(
    python scripts/sdr_stage_contract.py audit-lineage
    --base-model-manifest "${E_I_BASE_MODEL_MANIFEST}"
    --e-i-checkpoint "${E_I_CKPT}"
    --e-i-config "${E_I_CONFIG}"
    --dataset-stats "${DATASET_STATS}"
    --lineage-manifest "${E_I_LINEAGE_MANIFEST}"
    --out "${LINEAGE_AUDIT}"
)
run_command "${AUDIT_CMD[@]}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    LINEAGE_STATUS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${LINEAGE_AUDIT}")
    if [[ "${LINEAGE_STATUS}" != "PASS" ]]; then
        python scripts/sdr_stage_contract.py record-not-run \
            --lineage-audit "${LINEAGE_AUDIT}" \
            --validation-manifest "${SDR_VAL_MANIFEST}" \
            --warmstart-decision "${WARMSTART_DECISION}" \
            --repo "${PROJECT_REPO_ROOT}" \
            --output-dir "${RUN_DIR}"
        exit 3
    fi
fi

run_command python scripts/sdr_stage_contract.py check-lineage \
    --base-model-manifest "${E_I_BASE_MODEL_MANIFEST}" \
    --e-i-checkpoint "${E_I_CKPT}" \
    --e-i-config "${E_I_CONFIG}" \
    --dataset-stats "${DATASET_STATS}" \
    --lineage-manifest "${E_I_LINEAGE_MANIFEST}"
require_passed_decision "${WARMSTART_DECISION}"

CONFIG_OVERRIDES=(
    "task=${TASK}"
    "output_dir=${RUN_DIR}"
    "batch_size=1"
    "gradient_accumulation_steps=64"
    "mixed_precision=bf16"
    "learning_rate=1e-5"
    "data.train.pretrained_norm_stats=${DATASET_STATS}"
    "warm_start.kind=standalone_idm"
    "warm_start.checkpoint=${E_I_CKPT}"
    "warm_start.source_task=${SOURCE_TASK}"
    "warm_start.source_config=${E_I_CONFIG}"
    "warm_start.source_dataset_stats=${DATASET_STATS}"
    "dual_regime_training.optimizer.action_lr_scale=1.0"
    "dual_regime_training.optimizer.proprio_lr_scale=0.0"
    "dual_regime_training.optimizer.video_lr_scale=0.0"
)
resolve_hydra_config "${PROJECT_REPO_ROOT}/configs" train \
    "${RUN_DIR}/resolved_config.yaml" "${CONFIG_OVERRIDES[@]}"

PREFLIGHT_CMD=(
    python scripts/run_sdr_preflight.py
    --resolved-config "${RUN_DIR}/resolved_config.yaml"
    --e-i-checkpoint "${E_I_CKPT}"
    --e-i-config "${E_I_CONFIG}"
    --dataset-stats "${DATASET_STATS}"
    --lineage-manifest "${E_I_LINEAGE_MANIFEST}"
    --warmstart-decision "${WARMSTART_DECISION}"
    --validation-manifest "${SDR_VAL_MANIFEST}"
    --output-dir "${RUN_DIR}"
    --replays-per-mode 2
    --inference-steps 20
    --seed "${SEED}"
)
if [[ -n "${SIGMA_SHIFT:-}" ]]; then
    PREFLIGHT_CMD+=(--sigma-shift "${SIGMA_SHIFT}")
fi
run_command "${PREFLIGHT_CMD[@]}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    run_command python scripts/sdr_stage_contract.py check-probe \
        --preflight-decision "${RUN_DIR}/preflight_decision.json"
fi
