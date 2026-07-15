#!/usr/bin/env bash
# SETTING: Real Wan2.2 fused dual-regime forward versus the two-forward reference.
# MODEL/CHECKPOINT LINEAGE: Wan2.2 + ActionDiT construction only; no trained Gate.
# SCIENTIFIC GOAL: Prove the fused S-DR training graph preserves IDM/base numerics.
# ACCEPTANCE: Every test in test_dual_regime_fused.py passes with real-model tests enabled.
# REQUIRED INPUTS: Wan2.2/ActionDiT assets; optional FASTWAM_TEST_TASK and NPROC settings.
# OUTPUTS: Pytest report and run_manifest.json under EXPERIMENT_ROOT/P0.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"

FASTWAM_TEST_TASK=${FASTWAM_TEST_TASK:-libero_dual_regime_fused_2cam224_1e-4}
RUN_DIR="${EXPERIMENT_ROOT}/P0_fused_parity/${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
prepare_run_dir "${RUN_DIR}"
resolve_hydra_config "${PROJECT_REPO_ROOT}/configs" train \
    "${RUN_DIR}/resolved_config.yaml" "task=${FASTWAM_TEST_TASK}"
CMD=(
    env
    RUN_FASTWAM_MODEL_TESTS=1
    "FASTWAM_TEST_TASK=${FASTWAM_TEST_TASK}"
    "PYTHONPATH=${PROJECT_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    pytest -v tests/test_dual_regime_fused.py
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
)
run_command "${CMD[@]}"
run_command python "${DECISION_TOOL}" p0 \
    --check fused_real_wan_parity \
    --evidence tests/test_dual_regime_fused.py \
    --out "${RUN_DIR}/decision.json"
write_full_run_manifest "${RUN_DIR}/run_manifest.json"
