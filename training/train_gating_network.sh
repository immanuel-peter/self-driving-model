#!/bin/bash

# AutoMoE Gating Network Training Script
# This script trains the gating network using pre-trained expert models

set -eou pipefail

# System/runtime tuning for this machine (2x NVIDIA L40S, ~46GB each)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export TORCH_SHOW_CPP_STACKTRACES=1
# Single-node, NVLink present; disable IB to avoid NCCL timeouts if no IB
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

# Logging setup (mirror finetune_experts_carla.sh style)
mkdir -p logs
LOG_FILE="logs/train_gating_network_$(date +'%Y-%m-%d_%H-%M-%S').log"
{
  :
} > /dev/null
exec > >(tee "${LOG_FILE}") 2>&1
trap 'echo ""; echo "=============================================================="; echo "GATING TRAINING FAILED"; echo "End Time: $(date)"; echo "Log: ${LOG_FILE}"; echo "=============================================================="; exit 1' ERR

# Configuration
CONFIG_FILE="models/configs/gating_network/automoe_gating_v1/config.json"
DATA_ROOT="/ephemeral/datasets/carla/preprocessed"
CHECKPOINT_DIR="models/checkpoints/gating"
RUN_NAME="automoe_gating_v1"

# Expert checkpoint paths (update these with actual paths)
EXPERT_CHECKPOINTS=(
    # Order MUST match models/configs/automoe/model_config.json: detection, segmentation, drivable, nuscenes
    "models/checkpoints/carla_detection_expert_ddp/carla_det_ft_ddp/best.pth"
    "models/checkpoints/carla_segmentation_expert_ddp/carla_seg_ft_ddp/best.pth"
    "models/checkpoints/carla_drivable_expert_ddp/carla_drv_ft_ddp/best.pth"
    "models/checkpoints/carla_nuscenes_2d_ddp/nuscenes_img_only_carla_ft_ddp/best.pth"
)

########################################
# Training parameters (tuned for 2x L40S)
# Values are per-process (per-GPU when WORLD_SIZE>1)
BATCH_SIZE=32           # per GPU → effective 64
LEARNING_RATE=4e-4      # scaled up from 1e-4 for effective batch 64
EPOCHS=50               # faster convergence with larger batch
NUM_WORKERS=12          # total dataloader workers per process

# Multi-GPU training (set to 2 for this machine)
WORLD_SIZE=2

echo "=============================================================="
echo "STARTING GATING NETWORK TRAINING"
echo "Start Time: $(date)"
echo "Logging to: ${LOG_FILE}"
echo "Config: $CONFIG_FILE"
echo "Data: $DATA_ROOT"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Run name: $RUN_NAME"
echo "GPUs: ${WORLD_SIZE} | Per-GPU batch: ${BATCH_SIZE} | LR: ${LEARNING_RATE} | Epochs: ${EPOCHS}"
echo "=============================================================="
echo ""

# Check if data exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data directory $DATA_ROOT does not exist!"
    echo "Please ensure CARLA data is preprocessed and available."
    exit 1
fi

# Check if expert checkpoints exist
echo "Checking expert checkpoints..."
for checkpoint in "${EXPERT_CHECKPOINTS[@]}"; do
    if [ ! -f "$checkpoint" ]; then
        echo "Warning: Expert checkpoint $checkpoint not found!"
        echo "Training will proceed without pre-trained experts."
    else
        echo "Found: $checkpoint"
    fi
done
echo ""

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Update config with training parameters
echo "Updating configuration..."
python3 -c "
import json
import sys

# Load config
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

# Update training parameters
config['batch_size'] = $BATCH_SIZE
config['learning_rate'] = $LEARNING_RATE
config['epochs'] = $EPOCHS
config['num_workers'] = $NUM_WORKERS
config['run_name'] = '$RUN_NAME'

# Save updated config
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=4)

print('Configuration updated successfully!')
"

# Training command
if [ "$WORLD_SIZE" -eq 1 ]; then
    # Single GPU training
    echo "Starting single GPU training..."
    python3 training/train_gating_network.py \
        --config "$CONFIG_FILE" \
        --data_root "$DATA_ROOT" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --expert_checkpoints "${EXPERT_CHECKPOINTS[@]}"
else
    # Multi-GPU training (torchrun)
    echo "Starting multi-GPU training with $WORLD_SIZE GPUs..."
    torchrun --nproc_per_node="$WORLD_SIZE" --standalone \
        training/train_gating_network.py \
        --config "$CONFIG_FILE" \
        --data_root "$DATA_ROOT" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --expert_checkpoints "${EXPERT_CHECKPOINTS[@]}" \
        --world_size "$WORLD_SIZE"
fi

echo ""
echo "=============================================================="
echo "TRAINING COMPLETED"
echo "End Time: $(date)"
echo "Best model saved to: $CHECKPOINT_DIR/best.pth"
echo "Check tensorboard logs for training curves:"
echo "tensorboard --logdir models/runs/"
echo "Log: ${LOG_FILE}"
echo "=============================================================="

