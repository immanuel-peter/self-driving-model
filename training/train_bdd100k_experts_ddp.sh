#!/bin/bash
# train_bdd100k_experts_ddp.sh

LOG_FILE="bdd100k_experts_training_run_$(date +'%Y-%m-%d_%H-%M-%S').log"

EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.00001}
NUM_WORKERS=${NUM_WORKERS:-8}
DEVICE=${DEVICE:-cuda}
NUM_NODES=${NUM_NODES:-1}
NUM_GPUS=${NUM_GPUS:-2}

{
    echo "================================================================="
    echo "STARTING BDD100K EXPERT TRAINING PIPELINE"
    echo "Start Time: $(date)"
    echo "Logging all output to: ${LOG_FILE}"
    echo "================================================================="
    echo ""

    # --- 1. Detection Expert ---
    echo "[$(date)] ==> Starting Task 1/3: Detection Expert Training..."
    torchrec --nproc_per_node=2 python3 training/train_bdd100k_ddp.py \
        --task detection \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --run_name "detection_optimal_run" && \
    echo "[$(date)] ==> SUCCESS: Finished Task 1/3: Detection Expert Training."
    echo "-----------------------------------------------------------------"
    echo ""

    # --- 2. Drivable Area Expert ---
    echo "[$(date)] ==> Starting Task 2/3: Drivable Area Expert Training..."
    torchrec --nproc_per_node=2 python3 training/train_bdd100k_ddp.py \
        --task drivable \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --run_name "drivable_optimal_run" && \
    echo "[$(date)] ==> SUCCESS: Finished Task 2/3: Drivable Area Expert Training."
    echo "-----------------------------------------------------------------"
    echo ""

    # --- 3. Segmentation Expert ---
    echo "[$(date)] ==> Starting Task 3/3: Segmentation Expert Training..."
    torchrec --nproc_per_node=2 python3 training/train_bdd100k_ddp.py \
        --task segmentation \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --num_workers ${NUM_WORKERS} \
        --device ${DEVICE} \
        --run_name "segmentation_optimal_run" && \
    echo "[$(date)] ==> SUCCESS: Finished Task 3/3: Segmentation Expert Training."
    echo "-----------------------------------------------------------------"
    echo ""

    echo "================================================================="
    echo "TRAINING PIPELINE COMPLETED SUCCESSFULLY"
    echo "End Time: $(date)"
    echo "Full log saved to: ${LOG_FILE}"
    echo "================================================================="

} 2>&1 | tee "${LOG_FILE}"

