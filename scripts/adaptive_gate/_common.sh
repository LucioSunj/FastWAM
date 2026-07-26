#!/usr/bin/env bash
PROJECT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The outer workspace holds the shared launcher contract and every analyzer tool
# (MANIFEST_TOOL, DECISION_TOOL, ANALYZE_RESULTS_TOOL, ...). It defaults to the
# FastWAM parent directory, which is correct for the primary checkout. A linked
# worktree such as FastWAM-worktrees/<slug> has no such parent, so WORKSPACE_ROOT
# may be exported to point at the real outer repository instead.
#
# WORKSPACE_ROOT is exported rather than merely assigned because launchers spawn
# launchers: run_e1_sdr_learning_probe.sh `exec bash`-es run_e1_sdr_formal_train.sh
# and run_e1_train_shared_pilots.sh spawns run_e1_train_shared.sh. A child launcher
# must inherit the resolved root instead of re-deriving it from its own parent
# directory, which is wrong inside a worktree. Exporting here makes that guarantee
# independent of how the caller supplied the value -- in particular it covers a
# wrapper that assigns WORKSPACE_ROOT as a plain shell variable before sourcing.
# Nothing reads WORKSPACE_ROOT out of the environment (every consumer receives it
# as an explicit --workspace-root argument), so this is inert for the primary
# checkout, where a child would re-derive the same path anyway.
if [[ -n "${WORKSPACE_ROOT:-}" ]]; then
    if [[ ! -d "${WORKSPACE_ROOT}" ]]; then
        echo "WORKSPACE_ROOT is set but is not a directory: ${WORKSPACE_ROOT}" >&2
        return 2
    fi
else
    WORKSPACE_ROOT="$(cd "${PROJECT_REPO_ROOT}/.." && pwd)"
fi
export WORKSPACE_ROOT

_ADAPTIVE_GATE_SHARED_COMMON="${WORKSPACE_ROOT}/scripts/adaptive_gate/_common.sh"
if [[ ! -f "${_ADAPTIVE_GATE_SHARED_COMMON}" ]]; then
    echo "adaptive_gate launchers require the outer workspace scripts, but" >&2
    echo "    ${_ADAPTIVE_GATE_SHARED_COMMON}" >&2
    echo "does not exist." >&2
    echo "    PROJECT_REPO_ROOT=${PROJECT_REPO_ROOT}" >&2
    echo "    WORKSPACE_ROOT=${WORKSPACE_ROOT}" >&2
    echo "Fix: point WORKSPACE_ROOT at the outer repository root, for example" >&2
    echo "    export WORKSPACE_ROOT=/path/to/outer-repo" >&2
    echo "This is required in a linked worktree, whose parent directory is not the" >&2
    echo "outer repository." >&2
    unset _ADAPTIVE_GATE_SHARED_COMMON
    return 2
fi
source "${_ADAPTIVE_GATE_SHARED_COMMON}"
unset _ADAPTIVE_GATE_SHARED_COMMON
cd "${PROJECT_REPO_ROOT}"

require_selected_sdr_checkpoint() {
    local checkpoint=$1
    local selection=${2:-${S_DR_SELECTION:-}}
    [[ -n "${selection}" ]] || die "S_DR_SELECTION is required"
    require_file "${selection}"
    run_command python scripts/validate_sdr_checkpoint.py check-selection \
        --selection "${selection}" --checkpoint "${checkpoint}"
}

run_plus_endpoint_shards() {
    local method=$1
    local task=$2
    local checkpoint=$3
    local stats=$4
    local branch=$5
    local manifest=$6
    local output_dir=$7
    shift 7
    local num_shards=${NUM_SHARDS:-1}
    local cuda_devices=${CUDA_DEVICES:-0}
    IFS=',' read -r -a device_list <<< "${cuda_devices}"
    if [[ "${#device_list[@]}" -lt "${num_shards}" ]]; then
        die "CUDA_DEVICES provides ${#device_list[@]} devices for ${num_shards} shards"
    fi
    local pids=()
    local shard
    for ((shard = 0; shard < num_shards; shard++)); do
        local cmd=(
            env "CUDA_VISIBLE_DEVICES=${device_list[$shard]}"
            python scripts/evaluate_libero_plus_manifest.py
            --task "${task}"
            --ckpt "${checkpoint}"
            --dataset-stats "${stats}"
            --episode-manifest "${manifest}"
            --method "${method}"
            --force-branch "${branch}"
            --out "${output_dir}"
            --inference-steps "${INFERENCE_STEPS:-20}"
            --replan-steps "${REPLAN_STEPS:-10}"
            --wam-seed "${WAM_SEED:-0}"
            --shard-index "${shard}"
            --num-shards "${num_shards}"
        )
        if [[ -n "${SIGMA_SHIFT:-}" && "${SIGMA_SHIFT}" != "null" ]]; then
            cmd+=(--sigma-shift "${SIGMA_SHIFT}")
        fi
        cmd+=("$@")
        record_run_command "${cmd[@]}"
        print_command "${cmd[@]}"
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            :
        else
            "${cmd[@]}" &
            pids+=("$!")
        fi
    done
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        local pid
        local failed=0
        for pid in "${pids[@]}"; do
            if ! wait "${pid}"; then
                failed=1
            fi
        done
        [[ "${failed}" -eq 0 ]] || die "one or more endpoint evaluation shards failed"
        for ((shard = 0; shard < num_shards; shard++)); do
            RUN_ARTIFACTS+=(
                "${output_dir}/trials_shard_${shard}_of_${num_shards}.jsonl"
                "${output_dir}/summary_shard_${shard}_of_${num_shards}.json"
            )
        done
    fi
}

merge_endpoint_shards() {
    local output_dir=$1
    local method=$2
    local merged=$3
    local args=(python "${MERGE_JSONL_TOOL}" --out "${merged}")
    local shard
    for ((shard = 0; shard < ${NUM_SHARDS:-1}; shard++)); do
        args+=(--input "${output_dir}/trials_shard_${shard}_of_${NUM_SHARDS:-1}.jsonl")
    done
    run_command "${args[@]}"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        RUN_ARTIFACTS+=("${merged}")
    fi
}
