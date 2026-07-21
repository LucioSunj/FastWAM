#!/bin/bash

set -euo pipefail

task_list_file=${1:?"Usage: $0 /path/to/tasks.txt"}

require_non_empty() {
    local var_name="$1"
    local var_val="${!var_name:-}"
    if [ -z "$var_val" ]; then
        echo "Error: required variable $var_name is not set" >&2
        exit 1
    fi
}

ROOT_DIR=${ROOT_DIR:-"$(pwd)"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/evaluate_results/persistent_libero"}
SESSION_NAME=${SESSION_NAME:-"libero_persistent_workers"}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-"/root/autodl-fs/fastwam_libero_plus_eval/conda_envs/fastwam-libero-plus"}
CONDA_SH=${CONDA_SH:-"/root/miniconda3/bin/activate"}
EXTRA_ARGS=${EXTRA_ARGS:-}

require_non_empty "CONFIG"
require_non_empty "CKPT"
require_non_empty "NUM_TRIALS"

mkdir -p "$OUTPUT_DIR"
cp "$task_list_file" "$OUTPUT_DIR/tasks.txt" 2>/dev/null || true
task_list_file="$OUTPUT_DIR/tasks.txt"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    require_non_empty "NUM_GPUS"
    AVAILABLE_GPUS=$(seq 0 $((NUM_GPUS - 1)) | tr '\n' ',' | sed 's/,$//')
else
    AVAILABLE_GPUS=$CUDA_VISIBLE_DEVICES
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi
IFS=',' read -r -a GPU_ARRAY <<< "$AVAILABLE_GPUS"

QUEUE_FILE="$OUTPUT_DIR/persistent_queue.txt"
LOCK_FILE="$OUTPUT_DIR/persistent_queue.lock"
FAILED_TASKS_FILE="$OUTPUT_DIR/failed_tasks.txt"
FAILED_LOCK_FILE="$OUTPUT_DIR/persistent_failed_tasks.lock"
STATUS_DIR="$OUTPUT_DIR/persistent_worker_status"
LOG_DIR="$OUTPUT_DIR/persistent_worker_logs"
mkdir -p "$STATUS_DIR" "$LOG_DIR"
touch "$FAILED_TASKS_FILE"

python - "$task_list_file" "$OUTPUT_DIR" "$QUEUE_FILE" <<'PY'
import sys
from pathlib import Path

task_file = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
queue_file = Path(sys.argv[3])

pending = []
completed = 0
seen = set()

for raw in task_file.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line in seen:
        continue
    seen.add(line)
    suite, task_id_s = line.split(",", 1)
    task_id = int(task_id_s)
    if any((output_dir / suite).glob(f"gpu*_task{task_id}_results.json")):
        completed += 1
        continue
    pending.append(f"{suite},{task_id}")

queue_file.write_text("\n".join(pending) + ("\n" if pending else ""), encoding="utf-8")
print(f"Prepared persistent queue: pending={len(pending)} completed={completed} queue={queue_file}")
PY

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$SESSION_NAME"
fi

echo "Starting persistent LIBERO workers"
echo "Session: $SESSION_NAME"
echo "Output: $OUTPUT_DIR"
echo "GPUs: ${GPU_ARRAY[*]}"
echo "Queue: $QUEUE_FILE"

for idx in "${!GPU_ARRAY[@]}"; do
    gpu_id="${GPU_ARRAY[$idx]}"
    log_file="$LOG_DIR/gpu${gpu_id}.log"
    cmd_file="$LOG_DIR/gpu${gpu_id}.cmd.sh"
    cat > "$cmd_file" <<EOF
#!/bin/bash
set -euo pipefail
source "$CONDA_SH" "$CONDA_ENV_PATH"
cd "$ROOT_DIR"
export EXP_NAME="${EXP_NAME:-}"
export PERSISTENT_QUEUE_FILE="$QUEUE_FILE"
export PERSISTENT_LOCK_FILE="$LOCK_FILE"
export PERSISTENT_STATUS_DIR="$STATUS_DIR"
export PERSISTENT_FAILED_TASKS_FILE="$FAILED_TASKS_FILE"
export PERSISTENT_FAILED_LOCK_FILE="$FAILED_LOCK_FILE"
CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS MUJOCO_EGL_DEVICE_ID=0 python experiments/libero/eval_libero_persistent_worker.py \
  --config-name sim_libero_plus \
  task=$CONFIG \
  ckpt=$CKPT \
  gpu_id=$gpu_id \
  EVALUATION.device=cuda:$idx \
  EVALUATION.num_trials=$NUM_TRIALS \
  EVALUATION.output_dir=$OUTPUT_DIR \
  $EXTRA_ARGS \
  > "$log_file" 2>&1
EOF
    chmod +x "$cmd_file"
    if [ "$idx" -eq 0 ]; then
        tmux new-session -d -s "$SESSION_NAME" -n "gpu${gpu_id}" "bash '$cmd_file'"
    else
        tmux new-window -t "$SESSION_NAME" -n "gpu${gpu_id}" "bash '$cmd_file'"
    fi
done

echo "Persistent workers launched. Attach with: tmux attach -t $SESSION_NAME"
