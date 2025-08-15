#!/bin/bash
# training/train_bdd100k_and_nuscenes_finetune.sh

set -euo pipefail

# Ensure we run from repo root and have correct imports
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Optional skip flags (0/1, false/true, no/yes)
SKIP_TRAIN_BDD=${SKIP_TRAIN_BDD:-0}
SKIP_EVAL_BDD=${SKIP_EVAL_BDD:-0}
SKIP_TRAIN_NUSCENES=${SKIP_TRAIN_NUSCENES:-0}
SKIP_EVAL_NUSCENES=${SKIP_EVAL_NUSCENES:-0}

should_skip() {
	local v="${1:-0}"
	case "${v,,}" in
		1|true|yes) return 0 ;;
		*) return 1 ;;
	esac
}

# =========================
# BDD100K Detection Finetune
# =========================

# Tunables for BDD100K
BDD_RUN_NAME=${BDD_RUN_NAME:-"detection_a100_2x_ft"}
BDD_EPOCHS=${BDD_EPOCHS:-10}
BDD_BATCH_SIZE=${BDD_BATCH_SIZE:-24}
BDD_LR=${BDD_LR:-2e-4}
BDD_WD=${BDD_WD:-2e-5}
BDD_DEVICE=${BDD_DEVICE:-cuda}
BDD_NUM_WORKERS=${BDD_NUM_WORKERS:-12}
BDD_COST_CLASS=${BDD_COST_CLASS:-1.0}
BDD_COST_BBOX=${BDD_COST_BBOX:-5.0}
BDD_COST_GIOU=${BDD_COST_GIOU:-3.0}
BDD_BBOX_LOSS_W=${BDD_BBOX_LOSS_W:-8.0}
BDD_IMAGENET_NORM=${BDD_IMAGENET_NORM:-"--imagenet_norm"}
BDD_RESUME_FROM=${BDD_RESUME_FROM:-"models/checkpoints/bdd100k_detection_expert/detection_a100_2x/best.pth"}
BDD_RESUME_MODE=${BDD_RESUME_MODE:-"model"}
BDD_TASK=${BDD_TASK:-"detection"}
BDD_SPLIT=${BDD_SPLIT:-"val"}

# Logging
BDD_LOG_FILE="${BDD_RUN_NAME}_finetune_$(date +'%Y-%m-%d_%H-%M-%S').log"

if ! should_skip "$SKIP_TRAIN_BDD" || ! should_skip "$SKIP_EVAL_BDD"; then
  (
    set -euo pipefail
    exec > >(tee "${BDD_LOG_FILE}") 2>&1
    trap 'echo ""; echo "=============================================================="; echo "BDD100K SECTION FAILED"; echo "End Time: $(date)"; echo "Log: ${BDD_LOG_FILE}"; echo "=============================================================="; exit 1' ERR

    echo "=============================================================="
    echo "STARTING BDD100K DETECTION FINETUNE"
    echo "Start Time: $(date)"
    echo "Logging to: ${BDD_LOG_FILE}"
    echo "=============================================================="
    echo ""

    if ! should_skip "$SKIP_TRAIN_BDD"; then
      python training/train_bdd100k_ddp.py \
        --task ${BDD_TASK} \
        --run_name ${BDD_RUN_NAME} \
        --epochs ${BDD_EPOCHS} \
        --batch_size ${BDD_BATCH_SIZE} \
        --learning_rate ${BDD_LR} \
        --weight_decay ${BDD_WD} \
        --device ${BDD_DEVICE} \
        --num_workers ${BDD_NUM_WORKERS} \
        --cost_class ${BDD_COST_CLASS} \
        --cost_bbox ${BDD_COST_BBOX} \
        --cost_giou ${BDD_COST_GIOU} \
        --bbox_loss_weight ${BDD_BBOX_LOSS_W} \
        ${BDD_IMAGENET_NORM} \
        --resume_from ${BDD_RESUME_FROM} \
        --resume_mode ${BDD_RESUME_MODE}
    else
      echo "Skipping BDD100K training (SKIP_TRAIN_BDD=${SKIP_TRAIN_BDD})"
    fi

    if ! should_skip "$SKIP_EVAL_BDD"; then
      echo ""
      echo "=============================================================="
      echo "EVALUATING BDD100K DETECTION FINETUNE"
      echo "=============================================================="
      python eval/evaluate_bdd100k_expert.py \
        --task ${BDD_TASK} \
        --run_name ${BDD_RUN_NAME} \
        --split ${BDD_SPLIT} \
        --num_workers ${BDD_NUM_WORKERS} \
        ${BDD_IMAGENET_NORM}
    else
      echo "Skipping BDD100K evaluation (SKIP_EVAL_BDD=${SKIP_EVAL_BDD})"
    fi
  )
else
  echo "Skipping all BDD100K steps"
fi

# =========================
# NuScenes Finetune
# =========================

# Tunables for NuScenes
NUSC_RUN_NAME=${NUSC_RUN_NAME:-"nuscenes_ddp_opt_ft"}
NUSC_EPOCHS=${NUSC_EPOCHS:-10}
NUSC_BATCH_SIZE=${NUSC_BATCH_SIZE:-32}
NUSC_LR=${NUSC_LR:-1e-4}
NUSC_WD=${NUSC_WD:-2e-5}
NUSC_DEVICE=${NUSC_DEVICE:-cuda}
NUSC_NUM_WORKERS=${NUSC_NUM_WORKERS:-12}
NUSC_COST_CLASS=${NUSC_COST_CLASS:-1.0}
NUSC_COST_BBOX=${NUSC_COST_BBOX:-5.0}
NUSC_COST_GIOU=${NUSC_COST_GIOU:-4.0}
NUSC_BBOX_LOSS_W=${NUSC_BBOX_LOSS_W:-8.0}
NUSC_NUM_QUERIES=${NUSC_NUM_QUERIES:-100}
NUSC_RESUME_FROM=${NUSC_RESUME_FROM:-"models/checkpoints/nuscenes_expert/nuscenes_ddp_opt/best_model.pth"}
NUSC_RESUME_MODE=${NUSC_RESUME_MODE:-"model"}
NUSC_SPLIT=${NUSC_SPLIT:-"val"}

NUSC_LOG_FILE="${NUSC_RUN_NAME}_finetune_$(date +'%Y-%m-%d_%H-%M-%S').log"

if ! should_skip "$SKIP_TRAIN_NUSCENES" || ! should_skip "$SKIP_EVAL_NUSCENES"; then
  (
    set -euo pipefail
    exec > >(tee -a "${NUSC_LOG_FILE}") 2>&1
    trap 'echo ""; echo "=============================================================="; echo "NUSCENES SECTION FAILED"; echo "End Time: $(date)"; echo "Log: ${NUSC_LOG_FILE}"; echo "=============================================================="; exit 1' ERR

    echo ""
    echo "=============================================================="
    echo "STARTING NUSCENES EXPERT FINETUNE"
    echo "Start Time: $(date)"
    echo "Logging to: ${NUSC_LOG_FILE}"
    echo "=============================================================="
    echo ""

    if ! should_skip "$SKIP_TRAIN_NUSCENES"; then
      python training/train_nuscenes_expert_ddp.py \
        --run_name ${NUSC_RUN_NAME} \
        --epochs ${NUSC_EPOCHS} \
        --batch_size ${NUSC_BATCH_SIZE} \
        --learning_rate ${NUSC_LR} \
        --weight_decay ${NUSC_WD} \
        --device ${NUSC_DEVICE} \
        --num_workers ${NUSC_NUM_WORKERS} \
        --cost_class ${NUSC_COST_CLASS} \
        --cost_bbox ${NUSC_COST_BBOX} \
        --cost_giou ${NUSC_COST_GIOU} \
        --bbox_loss_weight ${NUSC_BBOX_LOSS_W} \
        --num_queries ${NUSC_NUM_QUERIES} \
        --resume_from ${NUSC_RESUME_FROM} \
        --resume_mode ${NUSC_RESUME_MODE}
    else
      echo "Skipping NuScenes training (SKIP_TRAIN_NUSCENES=${SKIP_TRAIN_NUSCENES})"
    fi

    if ! should_skip "$SKIP_EVAL_NUSCENES"; then
      echo ""
      echo "=============================================================="
      echo "EVALUATING NUSCENES EXPERT FINETUNE"
      echo "=============================================================="
      python eval/evaluate_nuscenes_expert.py \
        --run_name ${NUSC_RUN_NAME} \
        --split ${NUSC_SPLIT} \
        --num_workers ${NUSC_NUM_WORKERS}
    else
      echo "Skipping NuScenes evaluation (SKIP_EVAL_NUSCENES=${SKIP_EVAL_NUSCENES})"
    fi
  )
else
  echo "Skipping all NuScenes steps"
fi

echo ""
echo "=============================================================="
echo "ALL FINETUNE & EVALUATION COMPLETED"
echo "End Time: $(date)"
echo "BDD100K Log: ${BDD_LOG_FILE}"
echo "NuScenes Log: ${NUSC_LOG_FILE}"
echo "=============================================================="