## Training (Overview)

For full arguments and environment variables, open each script or run with `--help` when supported.

> Note: Feel free to modify these scripts to align with your development workflow.

---

### `train_bdd100k_experts_ddp.sh`
- **Purpose:** Train BDD100K experts (detection, drivable, segmentation) with multi-GPU DDP orchestration.
- **Inputs:**
  - Env: `NUM_GPUS`, `LEARNING_RATE`, `WEIGHT_DECAY`, `NUM_WORKERS`, `DEVICE`.
  - Flags: `SKIP_DETECTION`, `SKIP_DRIVABLE`, `SKIP_SEGMENTATION` to skip tasks.
- **Outputs:**
  - Checkpoints under `models/checkpoints/bdd100k_*_expert/<run_name>/best.pth`; logs in `logs/`.
- **Run:**
  ```bash
  bash training/train_bdd100k_experts_ddp.sh
  ```

---

### `train_nuscenes_expert_ddp.sh`
- **Purpose:** Train the nuScenes expert with DDP; configurable queries and loss weights.
- **Inputs:**
  - Env: `RUN_NAME`, `NUM_GPUS`, `EPOCHS`, `BATCH_SIZE`, `LR`, `WD`, `NUM_WORKERS`, `NUM_QUERIES`, `COST_*`, `BBOX_LOSS_W`.
- **Outputs:**
  - Checkpoints under `models/checkpoints/nuscenes_expert/<run_name>/`; logs in repo root or `logs/`.
- **Run:**
  ```bash
  bash training/train_nuscenes_expert_ddp.sh
  ```

---

### `train_bdd100k_ddp.py`
- **Purpose:** Single-task trainer for BDD100K (detection/drivable/segmentation) invoked by shell orchestration.
- **Inputs:**
  - Args: `--task {detection,drivable,segmentation}`, `--epochs`, `--batch_size`, `--learning_rate`, `--weight_decay`, `--num_workers`, `--device`, `--run_name`.
- **Outputs:**
  - Writes checkpoints/metrics as configured by parent script.
- **Run:**
  ```bash
  python3 training/train_bdd100k_ddp.py 
  ```

---

### `train_nuscenes_expert_ddp.py`
- **Purpose:** nuScenes expert trainer (image+LiDAR) with Hungarian matching and DETR-style heads.
- **Inputs:**
  - Args: standard training knobs (`--epochs`, `--batch_size`, `--learning_rate`, `--weight_decay`, `--num_workers`, `--num_queries`, cost weights).
- **Outputs:**
  - Checkpoints under `models/checkpoints/nuscenes_expert/<run_name>/`.
- **Run:**
  ```bash
  python3 training/train_nuscenes_expert_ddp.py 
  ```

---

### `finetune_experts_carla.sh`
- **Purpose:** Fine-tune BDD100K and nuScenes experts on CARLA domain using DDP.
- **Inputs:**
  - Env: `DATA_ROOT` (preprocessed CARLA dir), `NUM_GPUS`, `BATCH`, `WORKERS`, per-task epochs `EPOCHS_*`, skip flags `SKIP_*`.
- **Outputs:**
  - Checkpoints under `models/checkpoints/carla_*_expert_ddp/<run_name>/best.pth`; logs under `logs/`.
- **Run:**
  ```bash
  bash training/finetune_experts_carla.sh
  ```

---

### `train_carla_bdd_experts_ddp.py`
- **Purpose:** Task-specific trainer targeting CARLA data for (detection/drivable/segmentation) fine-tune.
- **Inputs:**
  - Args: `--task`, `--data_root`, `--epochs`, `--batch_size`, `--num_workers`, `--run_name`.
- **Outputs:**
  - Fine-tuned checkpoints under `models/checkpoints/carla_*_expert_ddp/`.
- **Run:**
  ```bash
  python3 training/train_carla_bdd_experts_ddp.py 
  ```

---

### `train_carla_nuscenes_expert_2d_ddp.py`
- **Purpose:** Fine-tune nuScenes 2D (image-only) expert on CARLA with configurable `--num_queries`.
- **Inputs:**
  - Args: `--data_root`, `--epochs`, `--batch_size`, `--num_workers`, `--num_queries`, `--run_name`.
- **Outputs:**
  - Fine-tuned checkpoints under `models/checkpoints/carla_nuscenes_2d_ddp/`.
- **Run:**
  ```bash
  python3 training/train_carla_nuscenes_expert_2d_ddp.py 
  ```

---

### `train_gating_network.sh`
- **Purpose:** Orchestrate gating network training (single or multi-GPU) with config updates and expert checkpoint checks.
- **Inputs:**
  - Files: `models/configs/gating_network/**/config.json` (updated in-place with run params).
  - Env: `WORLD_SIZE`, `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `NUM_WORKERS`.
  - Paths: `EXPERT_CHECKPOINTS` array to pre-trained experts.
- **Outputs:**
  - Gating checkpoints under `models/checkpoints/gating_network/<run>/best.pth`; tensorboard logs under `models/runs/`.
- **Run:**
  ```bash
  bash training/train_gating_network.sh
  ```

---

### `train_gating_network.py`
- **Purpose:** Python entrypoint for gating network training; supports DDP via torchrun.
- **Inputs:**
  - Args: `--config`, `--data_root`, `--checkpoint_dir`, `--expert_checkpoints [...]`, optional `--world_size`.
- **Outputs:**
  - Saves `best.pth` and logs under configured directories.
- **Run:**
  ```bash
  python3 training/train_gating_network.py 
  ```

---

### `train_carla_policy.py`
- **Purpose:** Train a CARLA control policy (trajectory/speed head) using preprocessed sequences.
- **Inputs:**
  - Args: typical training knobs and dataset paths (see file `--help`).
- **Outputs:**
  - Policy checkpoints under `models/checkpoints/carla_policy/`.
- **Run:**
  ```bash
  python3 training/train_carla_policy.py
  ```

---

### `hungarian_matcher.py`
- **Purpose:** Implementation utility for matching predictions to targets (used by detection trainers).
- **Use:** Imported by training/eval scripts; not run directly.


