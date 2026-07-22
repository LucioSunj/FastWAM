#!/usr/bin/env bash
# E1-P2: one preregistered 1e-5, 10-epoch S-DR run from E-I/S0.
set -euo pipefail
: "${EXPERIMENT_ROOT:=/root/autodl-tmp/experiments/adaptive_wm_reasoning}"
export EXPERIMENT_ROOT
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"

for name in \
    E_I_BASE_MODEL_MANIFEST E_I_CKPT E_I_CONFIG \
    E_I_LINEAGE_MANIFEST DATASET_STATS WARMSTART_DECISION SDR_VAL_MANIFEST \
    SDR_PREFLIGHT_DECISION SDR_LEARNING_PROBE_DECISION; do
    require_env "${name}"
    require_file "${!name}"
done

TASK=${TASK:-libero_dual_regime_fused_2cam224_1e-4}
SOURCE_TASK=${SOURCE_TASK:-libero_idm_2cam224_1e-4}
SEED=${SEED:-20260721}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="${EXPERIMENT_ROOT}/E1_P2/${RUN_ID}"
TRAIN_DIR="${RUN_DIR}/train"
BASELINE_DIR=$(dirname "${SDR_PREFLIGHT_DECISION}")
DISK_SAFETY_BYTES=${DISK_SAFETY_BYTES:-21474836480}
PRUNE_RECORD="${RUN_DIR}/pruned_artifacts.jsonl"
FORMAL_DECISION="${RUN_DIR}/formal_training_decision.json"
FORMAL_STARTED=0
prepare_run_dir "${RUN_DIR}"

record_interruption() {
    local exit_code=$?
    trap - ERR
    if [[ "${DRY_RUN}" -eq 0 && ! -f "${FORMAL_DECISION}" ]]; then
        local status=NOT-RUN
        if [[ "${FORMAL_STARTED}" -eq 1 ]]; then
            status=FAIL-DIAGNOSED
        fi
        python scripts/sdr_stage_contract.py record-formal-stop \
            --status "${status}" \
            --reason "formal launcher stopped with exit code ${exit_code}" \
            --preflight-decision "${SDR_PREFLIGHT_DECISION}" \
            --learning-probe-decision "${SDR_LEARNING_PROBE_DECISION}" \
            --lineage-manifest "${E_I_LINEAGE_MANIFEST}" \
            --out "${FORMAL_DECISION}" || true
    fi
    exit "${exit_code}"
}
trap record_interruption ERR

run_command python scripts/sdr_stage_contract.py check-lineage \
    --base-model-manifest "${E_I_BASE_MODEL_MANIFEST}" \
    --e-i-checkpoint "${E_I_CKPT}" \
    --e-i-config "${E_I_CONFIG}" \
    --dataset-stats "${DATASET_STATS}" \
    --lineage-manifest "${E_I_LINEAGE_MANIFEST}"
run_command python scripts/sdr_stage_contract.py check-formal \
    --preflight-decision "${SDR_PREFLIGHT_DECISION}" \
    --learning-probe-decision "${SDR_LEARNING_PROBE_DECISION}" \
    --lineage-manifest "${E_I_LINEAGE_MANIFEST}"

SCHEDULE=$(python scripts/sdr_stage_contract.py print-schedule \
    --preflight-decision "${SDR_PREFLIGHT_DECISION}")
E_I_SHA256=$(file_sha256 "${E_I_CKPT}")
BASE_OVERRIDES=(
    "task=${TASK}"
    "output_dir=${TRAIN_DIR}"
    "learning_rate=1e-5"
    "num_epochs=10"
    "max_steps=null"
    "run_until_step=null"
    "run_until_step_fraction=null"
    "batch_size=1"
    "gradient_accumulation_steps=64"
    "mixed_precision=bf16"
    "seed=${SEED}"
    "eval_every=-1"
    "save_every=-1"
    "save_steps=[]"
    "save_step_fractions=[]"
    "save_final_checkpoint=true"
    "save_optimizer_state=true"
    "weights_checkpoint_kind=action_dit_delta"
    "data.train.pretrained_norm_stats=${DATASET_STATS}"
    "+data.train.episode_split_manifest=${SDR_VAL_MANIFEST}"
    "+data.train.manifest_split=train"
    "warm_start.kind=standalone_idm"
    "warm_start.checkpoint=${E_I_CKPT}"
    "warm_start.expected_checkpoint_sha256=${E_I_SHA256}"
    "warm_start.source_task=${SOURCE_TASK}"
    "warm_start.source_config=${E_I_CONFIG}"
    "warm_start.source_dataset_stats=${DATASET_STATS}"
    "dual_regime_training.uncond_weight_schedule=${SCHEDULE}"
    "dual_regime_training.gradient_diagnostics_every=0"
    "dual_regime_training.optimizer.action_lr_scale=1.0"
    "dual_regime_training.optimizer.proprio_lr_scale=0.0"
    "dual_regime_training.optimizer.video_lr_scale=0.0"
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
)
resolve_hydra_config "${PROJECT_REPO_ROOT}/configs" train \
    "${RUN_DIR}/resolved_config.unplanned.yaml" "${BASE_OVERRIDES[@]}"
run_command python scripts/resolve_sdr_formal_steps.py \
    --input-config "${RUN_DIR}/resolved_config.unplanned.yaml" \
    --validation-manifest "${SDR_VAL_MANIFEST}" \
    --dataset-stats "${DATASET_STATS}" \
    --world-size 1 \
    --output-config "${RUN_DIR}/resolved_config.yaml" \
    --out "${RUN_DIR}/formal_step_contract.json"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    TOTAL_STEPS=$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["total_optimizer_steps"])' \
        "${RUN_DIR}/formal_step_contract.json")
else
    TOTAL_STEPS=${FORMAL_TOTAL_STEPS:-1000}
fi
if (( TOTAL_STEPS < 20 )); then
    die "formal step contract is too short for distinct monitoring fractions"
fi
run_command python scripts/sdr_stage_contract.py write-formal-manifest \
    --preflight-decision "${SDR_PREFLIGHT_DECISION}" \
    --learning-probe-decision "${SDR_LEARNING_PROBE_DECISION}" \
    --lineage-manifest "${E_I_LINEAGE_MANIFEST}" \
    --resolved-config "${RUN_DIR}/resolved_config.yaml" \
    --step-contract "${RUN_DIR}/formal_step_contract.json" \
    --validation-manifest "${SDR_VAL_MANIFEST}" \
    --warmstart-decision "${WARMSTART_DECISION}" \
    --launcher "${PROJECT_REPO_ROOT}/scripts/adaptive_gate/run_e1_sdr_formal_train.sh" \
    --repo "${PROJECT_REPO_ROOT}" \
    --outer-repo "${PROJECT_REPO_ROOT}/.." \
    --output-dir "${RUN_DIR}" \
    --disk-safety-bytes "${DISK_SAFETY_BYTES}" \
    --out "${RUN_DIR}/formal_run_manifest.json"

check_disk_floor() {
    run_command python -c \
        'import shutil,sys; p=sys.argv[1]; floor=int(sys.argv[2]); free=shutil.disk_usage(p).free; print({"free_bytes":free,"safety_floor_bytes":floor}); raise SystemExit(0 if free >= floor else 3)' \
        "${RUN_DIR}" "${DISK_SAFETY_BYTES}"
}

run_delta_diagnostics() {
    local delta=$1
    local output_dir=$2
    prepare_run_dir "${output_dir}"
    run_command python scripts/run_sdr_preflight.py \
        --resolved-config "${RUN_DIR}/resolved_config.yaml" \
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

prune_owned() {
    local kind=$1
    local path=$2
    run_command python scripts/prune_sdr_owned_artifact.py \
        --run-root "${RUN_DIR}" \
        --path "${path}" \
        --kind "${kind}" \
        --record "${PRUNE_RECORD}"
}

FRACTIONS=(0.05 0.10 0.25 0.50 0.75 1.00)
TAGS=(005 010 025 050 075 100)
MONITOR_ARGS=()
PREVIOUS_STATE=""
FINAL_DELTA=""
FINAL_DIAGNOSTICS=""
FORMAL_STARTED=1
for index in "${!FRACTIONS[@]}"; do
    fraction=${FRACTIONS[$index]}
    tag=${TAGS[$index]}
    target_step=$(python -c \
        'import math,sys; print(math.ceil(float(sys.argv[1])*int(sys.argv[2])))' \
        "${fraction}" "${TOTAL_STEPS}")
    printf -v step_tag '%06d' "${target_step}"
    check_disk_floor

    STAGE_OVERRIDES=(
        "${BASE_OVERRIDES[@]}"
        "max_steps=${TOTAL_STEPS}"
        "run_until_step=${target_step}"
    )
    if (( index > 0 )); then
        STAGE_OVERRIDES+=(
            "warm_start.kind=null"
            "warm_start.checkpoint=null"
            "warm_start.expected_checkpoint_sha256=null"
            "warm_start.source_task=null"
            "warm_start.source_config=null"
            "warm_start.source_dataset_stats=null"
            "resume=${PREVIOUS_STATE}"
        )
    fi
    if [[ "${fraction}" == "1.00" ]]; then
        STAGE_OVERRIDES+=("save_optimizer_state=false")
    fi
    resolve_hydra_config "${PROJECT_REPO_ROOT}/configs" train \
        "${RUN_DIR}/resolved_config.segment_${tag}.yaml" \
        "${STAGE_OVERRIDES[@]}"
    run_command env "RUN_ID=${RUN_ID}_${tag}" bash scripts/train_zero1.sh 1 \
        "${STAGE_OVERRIDES[@]}"

    delta="${TRAIN_DIR}/checkpoints/weights/step_${step_tag}.action_dit_delta.pt"
    require_file "${delta}"
    if [[ -n "${PREVIOUS_STATE}" ]]; then
        prune_owned state "${PREVIOUS_STATE}"
    fi
    diagnostics="${RUN_DIR}/diagnostics_fraction_${tag}"
    run_delta_diagnostics "${delta}" "${diagnostics}"
    MONITOR_ARGS+=(--evaluation "${fraction}=${diagnostics}")
    run_command python scripts/check_sdr_formal_monitor.py \
        --preflight-decision "${SDR_PREFLIGHT_DECISION}" \
        --baseline-diagnostics "${BASELINE_DIR}" \
        "${MONITOR_ARGS[@]}" \
        --training-metrics "${TRAIN_DIR}/training_metrics.jsonl" \
        --current-delta "${delta}" \
        --out "${RUN_DIR}/formal_monitor_${tag}.json"

    if [[ "${fraction}" == "0.05" || "${fraction}" == "0.25" || "${fraction}" == "0.75" ]]; then
        prune_owned delta "${delta}"
    fi
    if [[ "${fraction}" == "1.00" ]]; then
        FINAL_DELTA="${delta}"
        FINAL_DIAGNOSTICS="${diagnostics}"
        PREVIOUS_STATE=""
    else
        PREVIOUS_STATE="${TRAIN_DIR}/checkpoints/state/step_${step_tag}"
        require_dir "${PREVIOUS_STATE}"
    fi
done

FINAL_CHECKPOINT="${RUN_DIR}/fastwam_sdr_final.pt"
EXPECTED_FINAL_BYTES=$(python -c \
    'import os,sys; print(os.path.getsize(sys.argv[1]))' "${E_I_CKPT}")
run_command python -c \
    'import shutil,sys; p=sys.argv[1]; floor=int(sys.argv[2]); write=int(sys.argv[3]); free=shutil.disk_usage(p).free; print({"free_bytes":free,"required_before_reconstruction":floor+write}); raise SystemExit(0 if free >= floor+write else 3)' \
    "${RUN_DIR}" "${DISK_SAFETY_BYTES}" "${EXPECTED_FINAL_BYTES}"
run_command python scripts/reconstruct_sdr_checkpoint.py \
    --parent-checkpoint "${E_I_CKPT}" \
    --delta-checkpoint "${FINAL_DELTA}" \
    --output-checkpoint "${FINAL_CHECKPOINT}" \
    --decision-out "${RUN_DIR}/final_reconstruction.json"
run_command python scripts/validate_sdr_checkpoint.py validate \
    --checkpoint "${FINAL_CHECKPOINT}" \
    --resolved-config "${RUN_DIR}/resolved_config.yaml" \
    --dataset-stats "${DATASET_STATS}" \
    --expected-base-lr 1e-5 \
    --preflight-decision "${SDR_PREFLIGHT_DECISION}" \
    --out "${RUN_DIR}/formal_completion.json"
prune_owned delta "${FINAL_DELTA}"
run_command cp "${TRAIN_DIR}/training_metrics.jsonl" \
    "${RUN_DIR}/formal_training_metrics.jsonl"
run_command cp "${FINAL_DIAGNOSTICS}/gradient_diagnostics.json" \
    "${RUN_DIR}/formal_gradient_diagnostics.json"
run_command cp "${FINAL_DIAGNOSTICS}/generated_future_validation.json" \
    "${RUN_DIR}/formal_generated_future_validation.json"
run_command python scripts/decide_sdr_formal_training.py \
    --preflight-decision "${SDR_PREFLIGHT_DECISION}" \
    --learning-probe-decision "${SDR_LEARNING_PROBE_DECISION}" \
    --lineage-manifest "${E_I_LINEAGE_MANIFEST}" \
    --baseline-diagnostics "${BASELINE_DIR}" \
    --final-diagnostics "${FINAL_DIAGNOSTICS}" \
    --training-metrics "${RUN_DIR}/formal_training_metrics.jsonl" \
    --final-checkpoint "${FINAL_CHECKPOINT}" \
    --reconstruction-decision "${RUN_DIR}/final_reconstruction.json" \
    --checkpoint-completion "${RUN_DIR}/formal_completion.json" \
    --resolved-config "${RUN_DIR}/resolved_config.yaml" \
    --out "${FORMAL_DECISION}"
run_command python scripts/validate_sdr_checkpoint.py register-formal \
    --checkpoint "${FINAL_CHECKPOINT}" \
    --preflight-decision "${SDR_PREFLIGHT_DECISION}" \
    --learning-probe-decision "${SDR_LEARNING_PROBE_DECISION}" \
    --formal-training-decision "${FORMAL_DECISION}" \
    --lineage-manifest "${E_I_LINEAGE_MANIFEST}" \
    --out "${RUN_DIR}/s_dr_selection.json"
run_command python scripts/validate_sdr_checkpoint.py check-selection \
    --selection "${RUN_DIR}/s_dr_selection.json" \
    --checkpoint "${FINAL_CHECKPOINT}"
trap - ERR
