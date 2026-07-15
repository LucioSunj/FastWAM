#!/usr/bin/env bash
# SETTING: Offline WAM-Train action-agreement labels for the Recommended G-action baseline.
# MODEL/CHECKPOINT LINEAGE: frozen final S-DR -> paired UNCOND/IDM action errors; no E-U/E-I mixing.
# SCIENTIFIC GOAL: Build a cheap proxy baseline without presenting action agreement as causal uplift.
# ACCEPTANCE: Every shard passes strict checkpoint/stats/feature metadata checks and contains finite paired errors.
# REQUIRED INPUTS: SHARED_CKPT, S_DR_SELECTION, DATASET_STATS, COST_PROFILE, E1_DECISION; SHARD_INDEX/NUM_SHARDS select one disjoint shard.
# OUTPUTS: one versioned G-action tensor shard and run_manifest.json in the experiment run directory.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"
require_env SHARED_CKPT
require_env DATASET_STATS
require_env COST_PROFILE
require_env E1_DECISION
require_file "${SHARED_CKPT}"
require_file "${DATASET_STATS}"
require_file "${COST_PROFILE}"
require_passed_decision "${E1_DECISION}"
require_selected_sdr_checkpoint "${SHARED_CKPT}"

TASK=${TASK:-libero_dual_regime_fused_2cam224_1e-4}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_INDEX=${SHARD_INDEX:-0}
STRIDE=${STRIDE:-20}
EXEC_HORIZON=${EXEC_HORIZON:-10}
NUM_SEEDS=${NUM_SEEDS:-1}
SEED_BASE=${SEED_BASE:-0}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)_shard${SHARD_INDEX}}
RUN_DIR="${EXPERIMENT_ROOT}/G_action_labels/${RUN_ID}"
LABEL_DIR="${RUN_DIR}/labels"
prepare_run_dir "${LABEL_DIR}"
RUN_ARTIFACTS=(
    "${SHARED_CKPT}"
    "${DATASET_STATS}"
    "${COST_PROFILE}"
    "${E1_DECISION}"
    "${S_DR_SELECTION}"
)
resolve_hydra_config "${PROJECT_REPO_ROOT}/configs" train \
    "${RUN_DIR}/resolved_config.yaml" "task=${TASK}" \
    "eval_num_inference_steps=${INFERENCE_STEPS:-20}"

CMD=(
    python scripts/generate_gate_oracle_labels.py
    --task "${TASK}"
    --backbone-kind idm
    --ckpt "${SHARED_CKPT}"
    --dataset-stats "${DATASET_STATS}"
    --cost-table "${COST_PROFILE}"
    --stride "${STRIDE}"
    --num-shards "${NUM_SHARDS}"
    --shard-index "${SHARD_INDEX}"
    --exec-horizon "${EXEC_HORIZON}"
    --num-seeds "${NUM_SEEDS}"
    --seed-base "${SEED_BASE}"
    --out "${LABEL_DIR}"
)
if [[ -n "${SIGMA_SHIFT:-}" && "${SIGMA_SHIFT}" != "null" ]]; then
    CMD+=(--sigma-shift "${SIGMA_SHIFT}")
fi
CMD+=("${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}")
run_command "${CMD[@]}"
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
