#!/usr/bin/env bash
# E1-P1D-LC: 50-step low-cap (w_cap=0.09) shared S-DR Canary diagnostic.
#
# SETTING
#   Single-variable post-Canary diagnostic defined by
#   docs/sft_method_discussion/post_canary_low_cap_execution_plan.md. The only
#   training variable relative to the archived FAIL-DIAGNOSED E1-P1 Canary is the
#   UNCOND weight cap: 0.5 -> 0.09. Model, data, seed, LR, optimizer, trainable
#   groups, schedule fractions, run length and every conditioning contract are
#   held fixed.
#
# MODEL/CHECKPOINT LINEAGE
#   standalone E-I -> strict S0 warm-start -> 50 low-cap optimizer steps. The
#   failed Canary's step-50 ActionDiT delta is audit-only and is never an
#   initializer. E-U is not a permitted initializer.
#
# SCIENTIFIC GOAL
#   Decide whether lowering the UNCOND weight cap keeps improving UNCOND while
#   avoiding a negative common-noise IDM descent margin and measurable E-I
#   forgetting (H-scale), or whether the conflict survives a benign gradient
#   scale (H-structure).
#
# ACCEPTANCE
#   The 18 preregistered conditions in section 8 of the low-cap plan, evaluated
#   only on the step-50 checkpoint. Any failure is FAIL-DIAGNOSED. A PASS
#   authorizes planning a separately preregistered 500-step low-cap probe and
#   nothing else; this launcher always stops after writing its own decision.
#
# REQUIRED INPUTS
#   E_I_BASE_MODEL_MANIFEST, E_I_CKPT, E_I_CONFIG, E_I_LINEAGE_MANIFEST,
#   DATASET_STATS, WARMSTART_DECISION, SDR_VAL_MANIFEST,
#   GENERATED_FUTURE_CACHE_SOURCE, FAILED_CANARY_DECISION,
#   FRESH_SDR_PREFLIGHT_DECISION, LOW_CAP_PLAN, EXPERIMENT_ROOT.
#
# OUTPUTS
#   low_cap_preregistration.json, low_cap_resolved_config.yaml,
#   low_cap_training_metrics.jsonl, diagnostics_step{10,25,50}/,
#   low_cap_canary_decision.json, low_cap_run_manifest.json, SHA256SUMS.txt,
#   README.md.
set -euo pipefail
: "${EXPERIMENT_ROOT:=/root/autodl-tmp/experiments/adaptive_wm_reasoning}"
export EXPERIMENT_ROOT
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"

for name in \
    E_I_BASE_MODEL_MANIFEST E_I_CKPT E_I_CONFIG \
    E_I_LINEAGE_MANIFEST DATASET_STATS WARMSTART_DECISION SDR_VAL_MANIFEST \
    FAILED_CANARY_DECISION FRESH_SDR_PREFLIGHT_DECISION LOW_CAP_PLAN; do
    require_env "${name}"
    require_file "${!name}"
done
require_env GENERATED_FUTURE_CACHE_SOURCE
require_dir "${GENERATED_FUTURE_CACHE_SOURCE}"

DECIDE_TOOL="scripts/decide_sdr_low_cap_canary.py"

# (4) The E-I/data artifacts must still validate against the same lineage the
# failed Canary used.
run_command python scripts/sdr_stage_contract.py check-lineage \
    --base-model-manifest "${E_I_BASE_MODEL_MANIFEST}" \
    --e-i-checkpoint "${E_I_CKPT}" \
    --e-i-config "${E_I_CONFIG}" \
    --dataset-stats "${DATASET_STATS}" \
    --lineage-manifest "${E_I_LINEAGE_MANIFEST}"
# (1) A fresh E1-P0.5 PASS is required; a FAIL/NOT-RUN preflight stops here.
run_command python scripts/sdr_stage_contract.py check-decision \
    --schema fastwam-sdr-preflight-decision-v1 \
    --decision "${FRESH_SDR_PREFLIGHT_DECISION}"
require_passed_decision "${WARMSTART_DECISION}"

# (4) The generated-future cache must be the same verified creation run.
RESOLVED_CACHE_SOURCE=$(python scripts/sdr_stage_contract.py print-cache-source \
    --preflight-decision "${FRESH_SDR_PREFLIGHT_DECISION}")
EXPECTED_CACHE_SOURCE=$(cd "${GENERATED_FUTURE_CACHE_SOURCE}" && pwd)
if [[ "${RESOLVED_CACHE_SOURCE}" != "${EXPECTED_CACHE_SOURCE}" ]]; then
    die "fresh preflight cache source ${RESOLVED_CACHE_SOURCE} != ${EXPECTED_CACHE_SOURCE}"
fi

# (2,3,6) Resolve the locked low-cap schedule. This also enforces that the
# archived Canary is still FAIL-DIAGNOSED with the common-noise IDM margin
# failure, that its SHA256 is unchanged, that every invariant artifact binding
# matches, and that the fresh w0 stayed within 5% of the archived value.
SCHEDULE=$(python "${DECIDE_TOOL}" print-schedule \
    --preflight-decision "${FRESH_SDR_PREFLIGHT_DECISION}" \
    --failed-canary-decision "${FAILED_CANARY_DECISION}")

TASK=${TASK:-libero_dual_regime_fused_2cam224_1e-4}
SOURCE_TASK=${SOURCE_TASK:-libero_idm_2cam224_1e-4}
SEED=${SEED:-20260721}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="${EXPERIMENT_ROOT}/E1_P1D_LC/${RUN_ID}"
TRAIN_DIR="${RUN_DIR}/train"
BASELINE_DIR=$(dirname "${FRESH_SDR_PREFLIGHT_DECISION}")
PREREGISTRATION="${RUN_DIR}/low_cap_preregistration.json"
RESOLVED_CONFIG="${RUN_DIR}/low_cap_resolved_config.yaml"
DECISION="${RUN_DIR}/low_cap_canary_decision.json"
prepare_run_dir "${RUN_DIR}"
E_I_SHA256=$(file_sha256 "${E_I_CKPT}")

run_command python "${DECIDE_TOOL}" preregister \
    --preflight-decision "${FRESH_SDR_PREFLIGHT_DECISION}" \
    --failed-canary-decision "${FAILED_CANARY_DECISION}" \
    --low-cap-plan "${LOW_CAP_PLAN}" \
    --out "${PREREGISTRATION}"

# (5,6) Independent strict warm start from E-I/S0 with the serialized low-cap
# schedule. (7) Deltas are saved at steps 10, 25 and 50.
STAGE_OVERRIDES=(
    "task=${TASK}"
    "output_dir=${TRAIN_DIR}"
    "learning_rate=1e-5"
    "max_steps=50"
    "batch_size=1"
    "gradient_accumulation_steps=64"
    "mixed_precision=bf16"
    "eval_every=-1"
    "save_every=-1"
    "save_steps=[10,25,50]"
    "save_final_checkpoint=false"
    "save_optimizer_state=false"
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
    "${RESOLVED_CONFIG}" "${STAGE_OVERRIDES[@]}"
run_command env "RUN_ID=${RUN_ID}" bash scripts/train_zero1.sh 1 \
    "${STAGE_OVERRIDES[@]}"

# (8) Identical fixed diagnostics for every saved delta: same validation
# records, action seeds, generation cache, solver and normalization stats.
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
        --generated-future-cache-source "${GENERATED_FUTURE_CACHE_SOURCE}" \
        --output-dir "${output_dir}" \
        --replays-per-mode 2 \
        --inference-steps 20 \
        --seed "${SEED}" \
        --no-fail-exit
}

for step in 10 25 50; do
    printf -v step_tag '%06d' "${step}"
    delta="${TRAIN_DIR}/checkpoints/weights/step_${step_tag}.action_dit_delta.pt"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        require_file "${delta}"
    fi
    run_delta_diagnostics "${delta}" "${RUN_DIR}/diagnostics_step${step}"
done

# (9) One independent low-cap decision. Any acceptance failure is
# FAIL-DIAGNOSED; any provenance failure is NOT-RUN. The step-50 checkpoint is
# the only admissible outcome, so step 10/25 can never be re-selected.
run_command python "${DECIDE_TOOL}" decide \
    --preflight-decision "${FRESH_SDR_PREFLIGHT_DECISION}" \
    --failed-canary-decision "${FAILED_CANARY_DECISION}" \
    --low-cap-plan "${LOW_CAP_PLAN}" \
    --preregistration "${PREREGISTRATION}" \
    --resolved-config "${RESOLVED_CONFIG}" \
    --baseline-diagnostics "${BASELINE_DIR}" \
    --step10-diagnostics "${RUN_DIR}/diagnostics_step10" \
    --step25-diagnostics "${RUN_DIR}/diagnostics_step25" \
    --step50-diagnostics "${RUN_DIR}/diagnostics_step50" \
    --training-metrics "${TRAIN_DIR}/training_metrics.jsonl" \
    --step10-delta "${TRAIN_DIR}/checkpoints/weights/step_000010.action_dit_delta.pt" \
    --step25-delta "${TRAIN_DIR}/checkpoints/weights/step_000025.action_dit_delta.pt" \
    --step50-delta "${TRAIN_DIR}/checkpoints/weights/step_000050.action_dit_delta.pt" \
    --selected-step 50 \
    --repo "${PROJECT_REPO_ROOT}" \
    --out "${DECISION}"

run_command python "${DECIDE_TOOL}" write-evidence-index \
    --decision "${DECISION}" \
    --run-dir "${RUN_DIR}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    RUN_ARTIFACTS+=(
        "${PREREGISTRATION}"
        "${RESOLVED_CONFIG}"
        "${TRAIN_DIR}/training_metrics.jsonl"
        "${DECISION}"
        "${RUN_DIR}/SHA256SUMS.txt"
        "${RUN_DIR}/README.md"
    )
fi
write_full_run_manifest "${RUN_DIR}/low_cap_run_manifest.json"

# (10) E1-P1D-LC stops here whether it passed or failed. This launcher must
# never chain into the original E1-P1 probe route, the 500-step probe or the
# 10-epoch formal training route: a PASS only authorizes preregistering a
# separate 500-step low-cap probe as its own experiment.
