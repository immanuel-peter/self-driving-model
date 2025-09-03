## Evaluation Scripts (Overview)

For full arguments, run each script with `--help` or open the file.

> Note: Feel free to modify these scripts to align with your development workflow.

---

### `analyze_bdd100k_detection_batch.py`
- **Purpose**: Quick per-image analysis on BDD100K detection (matched IoU and recall@0.5) for the first N samples.
- **Loads**: `models/checkpoints/bdd100k_detection_expert/<run_name>/best.pth` and a BDD detection DataLoader.
- **Outputs**: Console table and JSON under `eval/results/` with per-image metrics.
- **Run**:
    ```bash
    python3 eval/analyze_bdd100k_detection_batch.py
    ```

---

### `evaluate_bdd100k_expert.py`
- **Purpose**: Evaluate BDD100K experts (detection, drivable, segmentation).
- **Loads**: Expert checkpoint by task and run name; corresponding BDD DataLoader.
- **Metrics**: 
    - Detection: val_loss, avg_iou, recall@0.5. 
    - Seg/Drivable: val_loss, pixel_acc, mean_iou.
- **Outputs**: JSON under `eval/results/` containing metrics and run metadata.
- **Run**:
    ```bash
    python3 eval/evaluate_bdd100k_expert.py
    ```

---

### `evaluate_gating_network.py`
- **Purpose**: Evaluate AutoMoE gating network on CARLA sequences; produce metrics and plots.
- **Loads**: Gating checkpoint, model config, CARLA sequence dataset.
- **Metrics**: ADE/FDE (L1 and Euclidean), speed loss, gating entropy; expert usage stats.
- **Outputs**: `evaluation_results.json` and plots (expert usage, training curves, context-expert correlations) in `eval/results/` or specified output dir.
- **Run**:
    ```bash
    python3 eval/evaluate_gating_network.py
    ```

---

### `evaluate_nuscenes_expert.py`
- **Purpose**: Evaluate the nuScenes expert (image+LiDAR, 2D queries).
- **Loads**: `models/checkpoints/nuscenes_expert/<run_name>/best_model.pth`; nuScenes DataLoader from preprocessed cache.
- **Metrics**: val_loss (classification + bbox).
- **Outputs**: JSON under `eval/results/` with metrics and run metadata.
- **Run**:
    ```bash
    python3 eval/evaluate_nuscenes_expert.py
    ```

---

### `visualize_bdd100k_detection.py`
- **Purpose**: Visualize BDD100K detection predictions vs. ground truth for a few batches.
- **Loads**: BDD detection checkpoint and DataLoader.
- **Outputs**: JPEGs under `eval/vis/` with GT (green) and predictions (red) overlays.
- **Run**:
    ```bash
    python3 eval/visualize_bdd100k_detection.py
    ```


