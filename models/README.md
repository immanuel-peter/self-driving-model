## Models (Overview)

This directory contains model definitions, configurations, and training artifacts for the AutoMoE system.

> Note: Feel free to modify these artifacts to align with your development workflow.

### `models/configs/`
- **Purpose:** JSON configurations for experts, CARLA fine-tuning, nuScenes expert, gating network, policy, and the integrated AutoMoE model.

### `models/experts/`
- **Purpose:** Expert networks for specific perception tasks.
- **Files:**
  - `bdd_detection_expert.py` — BDD100K object detection expert.
  - `bdd_drivable_expert.py` — BDD100K drivable area segmentation expert.
  - `bdd_segmentation_expert.py` — BDD100K semantic segmentation expert.
  - `nuscenes_expert.py` — nuScenes image(+optional LiDAR) detection expert with query-based head.
  - `expert_extractors.py` — Utilities to convert raw expert outputs into feature vectors for gating/policy.

### `models/gating/`
- **Purpose:** Gating module that weights expert features given context.
- **Files:**
  - `gating_network.py` — Gating network producing expert weights, combined features, and optional logits.

### `models/policy/`
- **Purpose:** Policy head that converts fused features into driving trajectories and speed.
- **Files:**
  - `trajectory_head.py` — Trajectory policy predicting waypoints and speed (sequence and/or last-step).

### `models/checkpoints/` and `models/runs/`
- `models/checkpoints/` — Saved model weights organized by component and run name. Each run typically contains `best.pth` (or similar) and may include training metadata.
- `models/runs/` — TensorBoard and auxiliary logs produced during training; used for visualization and experiment tracking.

### `automoe.py`
- Defines the `AutoMoE` class (full mixture-of-experts system):
  - Instantiates experts from config and loads `expert_extractors`.
  - Builds a context extractor and `gating_network`.
  - Attaches the `TrajectoryPolicy` as the policy head.
  - Forward returns predicted `waypoints`, `speed/speed_seq`, `expert_weights`, `context_features`, and fused `combined_features`.
- Includes utilities:
  - `create_automoe_model(config, device)` — factory for building the full model from JSON config.
  - `load_expert_checkpoints(paths)` — load pretrained expert weights (handles NuScenes variants).
  - `freeze_experts()` / `unfreeze_experts()` — control expert trainability during gating/policy training.
