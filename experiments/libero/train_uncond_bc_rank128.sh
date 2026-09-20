#!/usr/bin/env bash
# Launch one prepared BC configuration on four or eight explicitly assigned GPUs.
set -euo pipefail

variant="${1:?usage: bash train_uncond_bc_rank128.sh action|video_action [Hydra overrides]}"
shift
case "$variant" in
  action|video_action) ;;
  *) printf '%s\n' "Unknown BC variant: $variant" >&2; exit 2 ;;
esac
: "${CUDA_VISIBLE_DEVICES:?Set four or eight authorized GPU IDs}"
: "${FASTWAM_BC_OUTPUT_DIR:?Set a distinct output directory for this run}"
: "${DIFFSYNTH_MODEL_BASE_PATH:?Set the local DiffSynth model cache directory}"
IFS=',' read -r -a bc_devices <<< "$CUDA_VISIBLE_DEVICES"
bc_world_size=${#bc_devices[@]}
if [[ $bc_world_size -ne 4 && $bc_world_size -ne 8 ]]; then
  printf '%s\n' 'Rank-128 formal BC requires four or eight assigned GPUs.' >&2
  exit 2
fi

bc_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
bc_python="${FASTWAM_BC_PYTHON:-/home/amax/SJ/.conda/fastwam-libero-plus/bin/python}"
export PYTHONPATH="$bc_root/src${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$bc_python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$bc_world_size" \
  "$bc_root/experiments/libero/train_uncond_lora_bc.py" \
  "task=libero_uncond_lora_bc_${variant}_rank128" \
  "training.gradient_accumulation_steps=$((128 / bc_world_size))" "$@"
