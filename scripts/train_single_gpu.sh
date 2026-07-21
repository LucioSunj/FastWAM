#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train_single_gpu.sh task=<task_name> [hydra_overrides...]" >&2
  exit 2
fi

TASK_BASENAME="train"
for arg in "$@"; do
  if [[ "${arg}" == task=* ]]; then
    TASK_BASENAME="${arg#task=}"
    TASK_BASENAME="${TASK_BASENAME%.yaml}"
  fi
done

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_ROOT="${FASTWAM_RUN_ROOT:-/root/autodl-fs/fastwam/runs}"

exec accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero0.yaml \
  --num_processes 1 \
  scripts/train.py \
  "output_dir=${OUTPUT_ROOT}/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "$@"
