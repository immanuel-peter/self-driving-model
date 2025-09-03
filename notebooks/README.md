## Notebooks (Overview)

> Note: Feel free to modify these notebooks to align with your development workflow.

---

### `explore_carla_run.ipynb`
- Purpose: Inspect a single CARLA preprocessed run. Visualize frames, optional masks/boxes, basic trajectory stats, and verify camera settings (FOV/size). Useful for sanity checks before/after fine-tuning.
- Inputs (set in the first cell): path to `datasets/carla/preprocessed`, split (`train/val`), and a `run_*` folder ID.

---

### `explore_nuscenes.ipynb`
- Purpose: Inspect preprocessed nuScenes samples. Render images with boxes, peek at LiDAR points, and compute simple class distributions.
- Inputs (set in the first cell): path to `datasets/nuscenes/preprocessed` and split (`train/val`).

