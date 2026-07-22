#!/usr/bin/env bash
# E1-P1: 50-step Canary, then independent 500-step ActionDiT-only probe.
set -euo pipefail
: "${EXPERIMENT_ROOT:=/root/autodl-tmp/experiments/adaptive_wm_reasoning}"
export EXPERIMENT_ROOT
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"

for name in \
    WAN_ROBOT_BASE_CKPT WAN_ROBOT_BASE_CONFIG E_I_CKPT E_I_CONFIG \
    E_I_LINEAGE_MANIFEST DATASET_STATS WARMSTART_DECISION SDR_VAL_MANIFEST \
    SDR_PREFLIGHT_DECISION; do
    require_env "${name}"
    require_file "${!name}"
done

run_command python scripts/sdr_stage_contract.py check-lineage \
    --wan-robot-base-checkpoint "${WAN_ROBOT_BASE_CKPT}" \
    --wan-robot-base-config "${WAN_ROBOT_BASE_CONFIG}" \
    --e-i-checkpoint "${E_I_CKPT}" \
    --e-i-config "${E_I_CONFIG}" \
    --dataset-stats "${DATASET_STATS}" \
    --lineage-manifest "${E_I_LINEAGE_MANIFEST}"
run_command python scripts/sdr_stage_contract.py check-probe \
    --preflight-decision "${SDR_PREFLIGHT_DECISION}"

TASK=${TASK:-libero_dual_regime_fused_2cam224_1e-4}
SOURCE_TASK=${SOURCE_TASK:-libero_idm_2cam224_1e-4}
SEED=${SEED:-20260721}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="${EXPERIMENT_ROOT}/E1_P1/${RUN_ID}"
CANARY_DIR="${RUN_DIR}/canary"
PROBE_DIR="${RUN_DIR}/probe500"
BASELINE_DIR=$(dirname "${SDR_PREFLIGHT_DECISION}")
prepare_run_dir "${CANARY_DIR}"
prepare_run_dir "${PROBE_DIR}"
SCHEDULE=$(python scripts/sdr_stage_contract.py print-schedule \
    --preflight-decision "${SDR_PREFLIGHT_DECISION}")

common_overrides() {
    local output_dir=$1
    local max_steps=$2
    local save_steps=$3
    STAGE_OVERRIDES=(
        "task=${TASK}"
        "output_dir=${output_dir}"
        "learning_rate=1e-5"
        "max_steps=${max_steps}"
        "batch_size=1"
        "gradient_accumulation_steps=64"
        "mixed_precision=bf16"
        "eval_every=-1"
        "save_every=-1"
        "save_steps=${save_steps}"
        "save_final_checkpoint=false"
        "save_optimizer_state=false"
        "weights_checkpoint_kind=action_dit_delta"
        "data.train.pretrained_norm_stats=${DATASET_STATS}"
        "+data.train.episode_split_manifest=${SDR_VAL_MANIFEST}"
        "+data.train.manifest_split=train"
        "warm_start.kind=standalone_idm"
        "warm_start.checkpoint=${E_I_CKPT}"
        "warm_start.source_task=${SOURCE_TASK}"
        "warm_start.source_config=${E_I_CONFIG}"
        "warm_start.source_dataset_stats=${DATASET_STATS}"
        "dual_regime_training.uncond_weight_schedule=${SCHEDULE}"
        "dual_regime_training.gradient_diagnostics_every=0"
        "dual_regime_training.optimizer.action_lr_scale=1.0"
        "dual_regime_training.optimizer.proprio_lr_scale=0.0"
        "dual_regime_training.optimizer.video_lr_scale=0.0"
    )
}

run_stage_training() {
    local output_dir=$1
    local max_steps=$2
    local save_steps=$3
    common_overrides "${output_dir}" "${max_steps}" "${save_steps}"
    resolve_hydra_config "${PROJECT_REPO_ROOT}/configs" train \
        "${output_dir}/resolved_config.yaml" "${STAGE_OVERRIDES[@]}"
    run_command env "RUN_ID=${RUN_ID}" bash scripts/train_zero1.sh 1 \
        "${STAGE_OVERRIDES[@]}"
}

run_delta_diagnostics() {
    local delta=$1
    local output_dir=$2
    prepare_run_dir "${output_dir}"
    run_command python scripts/run_sdr_preflight.py \
        --resolved-config "${BASELINE_DIR}/resolved_config.yaml" \
        --e-i-checkpoint "${E_I_CKPT}" \
        --e-i-config "${E_I_CONFIG}" \
        --dataset-stats "${DATASET_STATS}" \
        --lineage-manifest "${E_I_LINEAGE_MANIFEST}" \
        --warmstart-decision "${WARMSTART_DECISION}" \
        --validation-manifest "${SDR_VAL_MANIFEST}" \
        --action-delta "${delta}" \
        --generated-future-cache-source "${BASELINE_DIR}" \
        --output-dir "${output_dir}" \
        --replays-per-mode 2 \
        --inference-steps 20 \
        --seed "${SEED}" \
        --no-fail-exit
}

# Canary is an independent S0 warm-start and never feeds the 500-step optimizer.
run_stage_training "${CANARY_DIR}/train" 50 "[50]"
CANARY_DELTA="${CANARY_DIR}/train/checkpoints/weights/step_000050.action_dit_delta.pt"
require_file "${CANARY_DELTA}"
run_delta_diagnostics "${CANARY_DELTA}" "${CANARY_DIR}/diagnostics_step50"
run_command python scripts/decide_sdr_learning_probe.py canary \
    --preflight-decision "${SDR_PREFLIGHT_DECISION}" \
    --baseline-diagnostics "${BASELINE_DIR}" \
    --canary-diagnostics "${CANARY_DIR}/diagnostics_step50" \
    --training-metrics "${CANARY_DIR}/train/training_metrics.jsonl" \
    --canary-delta "${CANARY_DELTA}" \
    --out "${CANARY_DIR}/canary_decision.json"

# The full probe restarts from the original E-I/S0, not from Canary.
run_stage_training "${PROBE_DIR}/train" 500 "[50,100,250,500]"
for step in 50 100 250 500; do
    printf -v step_tag '%06d' "${step}"
    delta="${PROBE_DIR}/train/checkpoints/weights/step_${step_tag}.action_dit_delta.pt"
    require_file "${delta}"
    run_delta_diagnostics "${delta}" "${PROBE_DIR}/diagnostics_step${step}"
    if [[ "${step}" -eq 50 || "${step}" -eq 100 ]]; then
        run_command python scripts/prune_sdr_owned_artifact.py \
            --run-root "${RUN_DIR}" \
            --path "${delta}" \
            --kind delta \
            --record "${PROBE_DIR}/pruned_artifacts.jsonl"
    fi
done
STEP500_DELTA="${PROBE_DIR}/train/checkpoints/weights/step_000500.action_dit_delta.pt"
run_command python scripts/decide_sdr_learning_probe.py probe \
    --preflight-decision "${SDR_PREFLIGHT_DECISION}" \
    --canary-decision "${CANARY_DIR}/canary_decision.json" \
    --step0-diagnostics "${BASELINE_DIR}" \
    --step50-diagnostics "${PROBE_DIR}/diagnostics_step50" \
    --step100-diagnostics "${PROBE_DIR}/diagnostics_step100" \
    --step250-diagnostics "${PROBE_DIR}/diagnostics_step250" \
    --step500-diagnostics "${PROBE_DIR}/diagnostics_step500" \
    --training-metrics "${PROBE_DIR}/train/training_metrics.jsonl" \
    --step500-delta "${STEP500_DELTA}" \
    --out "${RUN_DIR}/learning_probe_decision.json"

# A PASS decision automatically starts E1-P2 from E-I/S0, never from probe weights.
export SDR_LEARNING_PROBE_DECISION="${RUN_DIR}/learning_probe_decision.json"
exec bash scripts/adaptive_gate/run_e1_sdr_formal_train.sh
