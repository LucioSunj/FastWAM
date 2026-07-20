#!/usr/bin/env bash
# SETTING: Real Wan2.2 fused dual-regime forward versus the two-forward reference.
# MODEL/CHECKPOINT LINEAGE: Wan2.2 + ActionDiT construction only; no trained Gate.
# SCIENTIFIC GOAL: Prove the fused S-DR training graph preserves IDM/base numerics.
# ACCEPTANCE: Every test in test_dual_regime_fused.py passes with real-model tests enabled.
# REQUIRED INPUTS: Wan2.2/ActionDiT assets; optional FASTWAM_TEST_TASK and NPROC settings.
# OUTPUTS: JUnit, full pytest log, decision, GPU/timing evidence, and run manifest.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
parse_launcher_args "$@"

FASTWAM_TEST_TASK=${FASTWAM_TEST_TASK:-libero_dual_regime_fused_2cam224_1e-4}
RUN_DIR="${EXPERIMENT_ROOT}/P0_fused_parity/${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PYTEST_XML="${RUN_DIR}/pytest.xml"
PYTEST_LOG="${RUN_DIR}/pytest.log"
DECISION_JSON="${RUN_DIR}/decision.json"
GPU_MONITOR="${RUN_DIR}/gpu_monitor.csv"
GPU_BEFORE="${RUN_DIR}/nvidia_smi_before.txt"
GPU_AFTER="${RUN_DIR}/nvidia_smi_after.txt"
TIMING_JSON="${RUN_DIR}/timing.json"

prepare_run_dir "${RUN_DIR}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    : > "${PYTEST_LOG}"
fi

START_EPOCH=$(date +%s)
START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GPU_MONITOR_PID=""
PYTEST_STARTED=0
PYTEST_EXIT_CODE=125
RUNTIME_WRITTEN=0

stop_gpu_monitor() {
    if [[ -n "${GPU_MONITOR_PID}" ]]; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
        GPU_MONITOR_PID=""
    fi
}

write_runtime_evidence() {
    [[ "${DRY_RUN}" -eq 0 ]] || return 0
    [[ "${RUNTIME_WRITTEN}" -eq 0 ]] || return 0
    RUNTIME_WRITTEN=1
    stop_gpu_monitor
    nvidia-smi > "${GPU_AFTER}" 2>&1 || true
    local end_epoch end_utc elapsed peak
    end_epoch=$(date +%s)
    end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    elapsed=$((end_epoch - START_EPOCH))
    peak=$(awk -F',' '
        {
            value=$4
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            if (value ~ /^[0-9]+$/ && value + 0 > maximum) maximum=value + 0
        }
        END { if (maximum == "") print "null"; else print maximum }
    ' "${GPU_MONITOR}" 2>/dev/null || printf 'null\n')
    printf '{\n' > "${TIMING_JSON}"
    printf '  "dtype": "bfloat16",\n' >> "${TIMING_JSON}"
    printf '  "end_utc": "%s",\n' "${end_utc}" >> "${TIMING_JSON}"
    printf '  "peak_gpu_memory_mib": %s,\n' "${peak}" >> "${TIMING_JSON}"
    printf '  "start_utc": "%s",\n' "${START_UTC}" >> "${TIMING_JSON}"
    printf '  "wall_clock_seconds": %d\n' "${elapsed}" >> "${TIMING_JSON}"
    printf '}\n' >> "${TIMING_JSON}"
}

decision_command() {
    DECISION_CMD=(
        python "${DECISION_TOOL}" p0
        --check fused_real_wan_parity
        --pytest-junit "${PYTEST_XML}"
        --pytest-log "${PYTEST_LOG}"
        --pytest-exit-code "${PYTEST_EXIT_CODE}"
        --evidence tests/test_dual_regime_fused.py
        --out "${DECISION_JSON}"
    )
}

write_decision() {
    decision_command
    record_run_command "${DECISION_CMD[@]}"
    print_command "${DECISION_CMD[@]}"
    "${DECISION_CMD[@]}"
}

finalize_p0() {
    local launcher_exit=$?
    local finalization_exit=0
    trap - EXIT
    set +e
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        write_runtime_evidence || finalization_exit=$?
        if [[ ! -s "${DECISION_JSON}" ]]; then
            if [[ "${PYTEST_STARTED}" -eq 0 && "${launcher_exit}" -ne 0 ]]; then
                PYTEST_EXIT_CODE=${launcher_exit}
            fi
            write_decision || finalization_exit=$?
        fi
        local artifact
        for artifact in \
            "${PYTEST_XML}" "${PYTEST_LOG}" "${DECISION_JSON}" \
            "${GPU_MONITOR}" "${GPU_BEFORE}" "${GPU_AFTER}" "${TIMING_JSON}"; do
            if [[ -f "${artifact}" ]]; then
                RUN_ARTIFACTS+=("${artifact}")
            fi
        done
        write_full_run_manifest "${RUN_DIR}/run_manifest.json" || finalization_exit=$?
    fi
    if [[ "${launcher_exit}" -eq 0 && "${finalization_exit}" -ne 0 ]]; then
        launcher_exit=${finalization_exit}
    fi
    exit "${launcher_exit}"
}

if [[ "${DRY_RUN}" -eq 0 ]]; then
    trap finalize_p0 EXIT
    nvidia-smi > "${GPU_BEFORE}" 2>&1 || true
    (
        while true; do
            nvidia-smi \
                --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,driver_version \
                --format=csv,noheader,nounits || true
            sleep 1
        done
    ) > "${GPU_MONITOR}" 2>&1 &
    GPU_MONITOR_PID=$!
fi

resolve_hydra_config "${PROJECT_REPO_ROOT}/configs" train \
    "${RUN_DIR}/resolved_config.yaml" "task=${FASTWAM_TEST_TASK}"
CMD=(
    env
    RUN_FASTWAM_MODEL_TESTS=1
    "FASTWAM_TEST_TASK=${FASTWAM_TEST_TASK}"
    "PYTHONPATH=${PROJECT_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    pytest -v tests/test_dual_regime_fused.py
    "--junitxml=${PYTEST_XML}"
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
    run_command "${CMD[@]}"
    PYTEST_EXIT_CODE=0
    decision_command
    run_command "${DECISION_CMD[@]}"
    write_full_run_manifest "${RUN_DIR}/run_manifest.json"
    exit 0
fi

record_run_command "${CMD[@]}"
print_command "${CMD[@]}"
PYTEST_STARTED=1
set +e
"${CMD[@]}" 2>&1 | tee "${PYTEST_LOG}"
PYTEST_EXIT_CODE=${PIPESTATUS[0]}
set -e
write_runtime_evidence

set +e
write_decision
DECISION_EXIT=$?
set -e
FINAL_EXIT=${PYTEST_EXIT_CODE}
if [[ "${DECISION_EXIT}" -ne 0 ]]; then
    FINAL_EXIT=${DECISION_EXIT}
elif ! python -c '
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["status"] == "PASS" else 1)
' "${DECISION_JSON}"; then
    FINAL_EXIT=1
fi
exit "${FINAL_EXIT}"
