#!/bin/bash

# AutoMoE Inference Runner (CARLA closed-loop)

set -eou pipefail

cd /home/ubuntu/self-driving-model
source .venv/bin/activate

# System/runtime tuning (adjust as needed)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export TORCH_SHOW_CPP_STACKTRACES=1

# Defaults (can be overridden by env or CLI)
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-2000}
OUT_DIR=${OUT_DIR:-inference/results}
MODEL_CONFIG=${MODEL_CONFIG:-models/configs/automoe/model_config.json}
CHECKPOINT=${CHECKPOINT:-models/checkpoints/gating_network/automoe_gating_v2/best.pth}
TOWN=${TOWN:-Town10HD}

# Recording defaults (env overridable)
SAVE_FRAMES=${SAVE_FRAMES:-1}
EXPORT_GIF=${EXPORT_GIF:-1}
GIF_FPS=${GIF_FPS:-10}
GIF_SAMPLE_EVERY=${GIF_SAMPLE_EVERY:-1}
GIF_MAX_FRAMES=${GIF_MAX_FRAMES:-800}

usage() {
  cat <<USAGE
Usage: $0 --checkpoint ${CHECKPOINT} [--model_config ${MODEL_CONFIG}] [--town ${TOWN}] [--steps N] [--fixed_dt 0.05] [--width 800] [--height 600] [--fov 90] [--lookahead_m 3.0] [--kp 0.4 --ki 0.0 --kd 0.02] [--out_dir ${OUT_DIR}] [--host ${HOST}] [--port ${PORT}] [--] [extra args]

Environment overrides:
  OUT_DIR, HOST, PORT
  SAVE_FRAMES (0/1), EXPORT_GIF (0/1), GIF_FPS, GIF_SAMPLE_EVERY, GIF_MAX_FRAMES

Examples:
  $0 --checkpoint ${CHECKPOINT} \
     --model_config ${MODEL_CONFIG} --town ${TOWN} --steps 2000
USAGE
}

# Logging setup
mkdir -p logs
LOG_FILE="logs/run_automoe_$(date +'%Y-%m-%d_%H-%M-%S').log"
{
  :
} > /dev/null
exec > >(tee "${LOG_FILE}") 2>&1
trap 'echo ""; echo "=============================================================="; echo "INFERENCE FAILED"; echo "End Time: $(date)"; echo "Log: ${LOG_FILE}"; echo "=============================================================="; exit 1' ERR

echo "=============================================================="
echo "STARTING AUTOMOE INFERENCE"
echo "Start Time: $(date)"
echo "Logging to: ${LOG_FILE}"
echo "Out dir: ${OUT_DIR}"
echo "Town: ${TOWN}"
echo "Model config: ${MODEL_CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Host: ${HOST} | Port: ${PORT}"
echo "=============================================================="
echo ""

# Warn if CARLA Python API is not importable (non-fatal; script can still show help)
if ! python3 - <<'PY'
try:
  import carla  # noqa: F401
  print("CARLA Python API: OK")
except Exception as e:
  print(f"Warning: CARLA Python API not found or failed to import: {e}")
PY
then
  echo "(Continuing; runtime will likely no-op without CARLA)"
fi

mkdir -p "${OUT_DIR}"

# Build recording flags
REC_ARGS=()
if [[ "${SAVE_FRAMES}" == "1" ]]; then
  REC_ARGS+=(--save_frames)
fi
if [[ "${EXPORT_GIF}" == "1" ]]; then
  REC_ARGS+=(--export_gif --gif_fps "${GIF_FPS}" --gif_sample_every "${GIF_SAMPLE_EVERY}" --gif_max_frames "${GIF_MAX_FRAMES}")
fi

echo "Launching Python..."
python3 inference/run_automoe.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --checkpoint "${CHECKPOINT}" \
  --model_config "${MODEL_CONFIG}" \
  --town "${TOWN}" \
  --out_dir "${OUT_DIR}" \
  "${REC_ARGS[@]}" \
  "$@"

STATUS=$?
echo ""
echo "=============================================================="
if [[ ${STATUS} -eq 0 ]]; then
  echo "INFERENCE COMPLETED"
  echo "End Time: $(date)"
  echo "Results dir: ${OUT_DIR}"
  echo "Check logs and optional GIF in results folder."
  echo "Log: ${LOG_FILE}"
else
  echo "INFERENCE FAILED (exit ${STATUS})"
  echo "End Time: $(date)"
  echo "Log: ${LOG_FILE}"
  exit ${STATUS}
fi
echo "=============================================================="


