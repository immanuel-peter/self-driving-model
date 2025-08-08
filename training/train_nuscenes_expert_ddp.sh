#!/bin/bash
# training/train_nuscenes_expert_ddp.sh
# DDP launch script for NuScenes expert with optimal config for 2×A100

set -euo pipefail

RUN_NAME=${RUN_NAME:-"nuscenes_ddp_opt"}
LOG_FILE="nuscenes_${RUN_NAME}_$(date +'%Y-%m-%d_%H-%M-%S').log"

# Tunables for your VM (2×A100 80GB, 24 CPUs, NVMe)
NUM_NODES=${NUM_NODES:-1}
NUM_GPUS=${NUM_GPUS:-2}
EPOCHS=${EPOCHS:-50}
# Batch size is per-GPU in our DDP code
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-1e-4}
WD=${WD:-1e-5}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_QUERIES=${NUM_QUERIES:-100}
COST_CLASS=${COST_CLASS:-1.0}
COST_BBOX=${COST_BBOX:-5.0}
COST_GIOU=${COST_GIOU:-2.0}
BBOX_LOSS_W=${BBOX_LOSS_W:-5.0}

{
  echo "=============================================================="
  echo "STARTING NUSCENES EXPERT TRAINING (DDP)"
  echo "Start Time: $(date)"
  echo "Logging to: ${LOG_FILE}"
  echo "GPUs: ${NUM_GPUS} | Epochs: ${EPOCHS} | Per-GPU batch: ${BATCH_SIZE}"
  echo "=============================================================="
  echo ""

  torchrun --nproc_per_node=${NUM_GPUS} --standalone \
    training/train_nuscenes_expert_ddp.py \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --learning_rate ${LR} \
      --weight_decay ${WD} \
      --device cuda \
      --num_workers ${NUM_WORKERS} \
      --run_name "${RUN_NAME}" \
      --cost_class ${COST_CLASS} \
      --cost_bbox ${COST_BBOX} \
      --cost_giou ${COST_GIOU} \
      --bbox_loss_weight ${BBOX_LOSS_W} \
      --num_queries ${NUM_QUERIES}

  echo ""
  echo "=============================================================="
  echo "TRAINING COMPLETED"
  echo "End Time: $(date)"
  echo "Log: ${LOG_FILE}"
  echo "=============================================================="
} 2>&1 | tee "${LOG_FILE}"
