#!/bin/bash
# training/finetune_experts_carla.sh

set -euo pipefail

# Tunables / defaults
NUM_GPUS=${NUM_GPUS:-4}
DATA_ROOT=${DATA_ROOT:-/ephemeral/datasets/carla/preprocessed}
DET_RUN=${DET_RUN:-carla_det_ft_ddp}
DRV_RUN=${DRV_RUN:-carla_drv_ft_ddp}
SEG_RUN=${SEG_RUN:-carla_seg_ft_ddp}
NUSC_RUN=${NUSC_RUN:-nuscenes_img_only_carla_ft_ddp}
EPOCHS_DET=${EPOCHS_DET:-20}
EPOCHS_DRV=${EPOCHS_DRV:-20}
EPOCHS_SEG=${EPOCHS_SEG:-20}
EPOCHS_NUSC=${EPOCHS_NUSC:-10}
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-8}
NUM_QUERIES=${NUM_QUERIES:-196}

# Skip flags (set to 1 to skip, 0 to run)
SKIP_DETECTION=${SKIP_DETECTION:-0}
SKIP_DRIVABLE=${SKIP_DRIVABLE:-0}
SKIP_SEGMENTATION=${SKIP_SEGMENTATION:-0}
SKIP_NUSCENES=${SKIP_NUSCENES:-0}

# CUDA / NCCL optimizations
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True,max_split_size_mb:128"}

LOG_FILE="logs/finetune_experts_carla_$(date +'%Y-%m-%d_%H-%M-%S').log"
{
  :
} > /dev/null
exec > >(tee "${LOG_FILE}") 2>&1

trap 'echo ""; echo "=============================================================="; echo "FINETUNE PIPELINE FAILED"; echo "End Time: $(date)"; echo "Log: ${LOG_FILE}"; echo "Stopping carla-policy with brev..."; brev stop carla-policy; echo "=============================================================="; exit 1' ERR

echo "=============================================================="
echo "STARTING CARLA FINETUNE PIPELINE (DDP)"
echo "Start Time: $(date)"
echo "Logging to: ${LOG_FILE}"
echo "GPUs: ${NUM_GPUS} | Per-GPU batch: ${BATCH}"
echo "Data root: ${DATA_ROOT}"
echo "=============================================================="
echo ""

if [[ "${SKIP_DETECTION}" != "1" ]]; then
  echo "[1/4] BDD Detection → CARLA"
  echo "START detection: $(date)"
  torchrun --nproc_per_node=${NUM_GPUS} --standalone training/train_carla_bdd_experts_ddp.py \
    --task detection \
    --data_root ${DATA_ROOT} \
    --epochs ${EPOCHS_DET} --batch_size ${BATCH} --num_workers ${WORKERS} \
    --run_name ${DET_RUN}
  echo "END detection: $(date)"
  echo ""
else
  echo "[1/4] BDD Detection → CARLA: SKIPPED"
  echo ""
fi

if [[ "${SKIP_DRIVABLE}" != "1" ]]; then
  echo "[2/4] BDD Drivable → CARLA"
  echo "START drivable: $(date)"
  torchrun --nproc_per_node=${NUM_GPUS} --standalone training/train_carla_bdd_experts_ddp.py \
    --task drivable \
    --data_root ${DATA_ROOT} \
    --epochs ${EPOCHS_DRV} --batch_size ${BATCH} --num_workers ${WORKERS} \
    --run_name ${DRV_RUN}
  echo "END drivable: $(date)"
  echo ""
else
  echo "[2/4] BDD Drivable → CARLA: SKIPPED"
  echo ""
fi

if [[ "${SKIP_SEGMENTATION}" != "1" ]]; then
  echo "[3/4] BDD Segmentation → CARLA"
  echo "START segmentation: $(date)"
  torchrun --nproc_per_node=${NUM_GPUS} --standalone training/train_carla_bdd_experts_ddp.py \
    --task segmentation \
    --data_root ${DATA_ROOT} \
    --epochs ${EPOCHS_SEG} --batch_size ${BATCH} --num_workers ${WORKERS} \
    --run_name ${SEG_RUN}
  echo "END segmentation: $(date)"
  echo ""
else
  echo "[3/4] BDD Segmentation → CARLA: SKIPPED"
  echo ""
fi

if [[ "${SKIP_NUSCENES}" != "1" ]]; then
  echo "[4/4] NuScenes (image-only 2D) → CARLA"
  echo "START nusc_2d: $(date)"
  torchrun --nproc_per_node=${NUM_GPUS} --standalone training/train_carla_nuscenes_expert_2d_ddp.py \
    --data_root ${DATA_ROOT} \
    --epochs ${EPOCHS_NUSC} --batch_size ${BATCH} --num_workers ${WORKERS} --num_queries ${NUM_QUERIES} \
    --run_name ${NUSC_RUN}
  echo "END nusc_2d: $(date)"
else
  echo "[4/4] NuScenes (image-only 2D) → CARLA: SKIPPED"
fi

echo ""
echo "=============================================================="
echo "FINETUNE PIPELINE COMPLETED"
echo "End Time: $(date)"
echo "Log: ${LOG_FILE}"
echo "=============================================================="