#!/usr/bin/env bash
PROJECT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_REPO_ROOT}/../scripts/adaptive_gate/_common.sh"
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
