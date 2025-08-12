#!/bin/bash
# train_bdd100k_experts_ddp.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

LOG_FILE="bdd100k_experts_training_run_$(date +'%Y-%m-%d_%H-%M-%S').log"

EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-16}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
NUM_WORKERS=${NUM_WORKERS:-12}
DEVICE=${DEVICE:-cuda}
NUM_NODES=${NUM_NODES:-1}
NUM_GPUS=${NUM_GPUS:-2}

{
    :
} > /dev/null

# Configure logging so set -e works. Don't pipe whole block; redirect FD instead.
exec > >(tee "${LOG_FILE}") 2>&1

echo "================================================================="
echo "STARTING BDD100K EXPERT TRAINING PIPELINE"
echo "Start Time: $(date)"
echo "Logging all output to: ${LOG_FILE}"
echo "================================================================="
echo ""

    # --- 1. Detection Expert ---
if [[ "${SKIP_DETECTION:-0}" != "1" ]]; then
    echo "[$(date)] ==> Starting Task 1/3: Detection Expert Training..."
    torchrun --nproc_per_node=${NUM_GPUS} --standalone training/train_bdd100k_ddp.py \
        --task detection \
        --epochs 60 \
        --batch_size 16 \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --run_name "detection_a100_2x"
    echo "[$(date)] ==> SUCCESS: Finished Task 1/3: Detection Expert Training."
    echo "-----------------------------------------------------------------"
    echo ""
else
    echo "[$(date)] ==> Skipping Task 1/3: Detection Expert (SKIP_DETECTION=1)."
    echo "-----------------------------------------------------------------"
    echo ""
fi

    # --- 2. Drivable Area Expert ---
if [[ "${SKIP_DRIVABLE:-0}" != "1" ]]; then
    echo "[$(date)] ==> Starting Task 2/3: Drivable Area Expert Training..."
    torchrun --nproc_per_node=${NUM_GPUS} --standalone training/train_bdd100k_ddp.py \
        --task drivable \
        --epochs 50 \
        --batch_size 48 \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --run_name "drivable_a100_2x"
    echo "[$(date)] ==> SUCCESS: Finished Task 2/3: Drivable Area Expert Training."
    echo "-----------------------------------------------------------------"
    echo ""
else
    echo "[$(date)] ==> Skipping Task 2/3: Drivable Area Expert (SKIP_DRIVABLE=1)."
    echo "-----------------------------------------------------------------"
    echo ""
fi

    # --- 3. Segmentation Expert ---
if [[ "${SKIP_SEGMENTATION:-0}" != "1" ]]; then
    echo "[$(date)] ==> Starting Task 3/3: Segmentation Expert Training..."
    torchrun --nproc_per_node=${NUM_GPUS} --standalone training/train_bdd100k_ddp.py \
        --task segmentation \
        --epochs 60 \
        --batch_size 32 \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --run_name "segmentation_a100_2x"
    echo "[$(date)] ==> SUCCESS: Finished Task 3/3: Segmentation Expert Training."
    echo "-----------------------------------------------------------------"
    echo ""
else
    echo "[$(date)] ==> Skipping Task 3/3: Segmentation Expert (SKIP_SEGMENTATION=1)."
    echo "-----------------------------------------------------------------"
    echo ""
fi

echo "================================================================="
echo "TRAINING PIPELINE COMPLETED SUCCESSFULLY"
echo "End Time: $(date)"
echo "Full log saved to: ${LOG_FILE}"
echo "================================================================="


