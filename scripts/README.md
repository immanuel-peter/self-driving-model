## Scripts (Overview)

For full arguments and environment variables, open each script or run with `--help` when supported.

> Note: Feel free to modify these scripts to align with your development workflow.

---

### `check_nuscenes.py`
- **Purpose:** Sanity-check nuScenes raw dataset availability; scans scenes and verifies camera file presence.
- **Inputs:**
  - Env: `NUSC_VERSION` (default `v1.0-trainval`), `NUSC_DATAROOT` (default `datasets/nuscenes/raw`).
- **Outputs:**
  - Console summary of available scenes; no files written.
- **Run:**
  ```bash
  python3 scripts/check_nuscenes.py
  ```

---

### `collect_autopilot_data_old.py`
- **Purpose:** Legacy CARLA autopilot data collection (RGB multi-camera) in synchronous mode.
- **Inputs:**
  - Requires CARLA server on `localhost:2000` and Python API.
  - Args: `--runs`, `--duration`, `--vehicles`, `--walkers`, `--output`, `--save-every`, etc.
  - Env: `CARLA_DATA_PATH` (default `~/automoe_training`).
- **Outputs:**
  - `run_*/images/<front|front_left|front_right|rear>/*.png`, `vehicle_log.json`, `collisions.json`, `config.json`.
- **Notes:** Multi-cam RGB only; suitable for quick baseline data.
- **Run:**
  ```bash
  python3 scripts/collect_autopilot_data_old.py
  ```

---

### `collect_autopilot_data.py`
- **Purpose:** CARLA autopilot data collection with RGB (multi-cam), semantic segmentation (front), LiDAR, and 2D boxes; logs vehicle state and traffic context.
- **Inputs:**
  - Requires CARLA server on `localhost:2000` and Python API.
  - Args: `--runs`/`--single-run`, `--continue-from`, `--duration`, `--vehicles`, `--walkers`, `--save-every`, `--output`.
  - Env: `CARLA_DATA_PATH` (default `~/self-driving-model/datasets/carla/raw`).
- **Outputs:**
  - `images/<cam>/*.png`, `segmentation/front/*.png` (+ colorized `segmentation_vis/front/*.png`), `lidar/*.npy` (Nx4), `annots/front/*.json` (2D boxes), `vehicle_log.json`, `collisions.json`, `config.json`.
- **Notes:** Synchronous mode with warm-up; front camera FOV=90; saves intrinsics-consistent artifacts.
- **Run:**
  ```bash
  python3 scripts/collect_autopilot_data.py
  ```

---

### `download_waymo_e2e_subset.py`
- **Purpose:** Download a subset of Waymo End-to-End camera TFRecords from GCS via `gsutil`.
- **Inputs:**
  - Requires `gsutil` installed and authenticated.
  - Args: `--bucket`, `--train`, `--val`, `--random`, `--outdir`.
- **Outputs:**
  - TFRecords under `<outdir>/train` and `<outdir>/val`; includes `val_sequence_name_to_scenario_cluster.json`.
- **Run:**
  ```bash
  python3 scripts/download_waymo_e2e_subset.py
  ```

---

### `preprocess_bdd100k.py`
- **Purpose:** Convert BDD100K raw images/labels into cached `.pt` files for detection, drivable, or segmentation.
- **Inputs:**
  - Args: `--task {detection,drivable,segmentation}`, `--raw_dir`, `--out_dir`.
  - Expects BDD100K folder structure under `raw_dir` (images/100k or 10k; labels/*).
- **Outputs:**
  - `datasets/bdd100k/preprocessed/<task>/<split>/*.pt` with image paths and task-specific targets.
- **Run:**
  ```bash
  python3 scripts/preprocess_bdd100k.py
  ```

---

### `preprocess_carla.py`
- **Purpose:** Convert CARLA raw runs into `.pt` frames with image, optional mask/boxes/lidar, intrinsics, vehicle state, and context; split into train/val.
- **Inputs:**
  - Args: `--raw_dir`, `--out_dir`, `--runs` (optional list), `--split_ratio`.
  - Expects raw `run_*/` folders from collection scripts.
- **Outputs:**
  - `datasets/carla/preprocessed/{train,val}/run_*/<frame>.pt` containing standardized tensors and metadata.
- **Run:**
  ```bash
  python3 scripts/preprocess_carla.py
  ```

---

### `preprocess_nuscenes.py`
- **Purpose:** Build nuScenes preprocessed cache: image, LiDAR, `Box` objects, and camera intrinsics by split.
- **Inputs:**
  - Env: `NUSC_VERSION`, `NUSC_DATAROOT`.
  - Requires `nuscenes-devkit`.
- **Outputs:**
  - `datasets/nuscenes/preprocessed/<split>/*.pt` per sample token.
- **Run:**
  ```bash
  python3 scripts/preprocess_nuscenes.py
  ```

---

### `redo_preprocess.py`
- **Purpose:** Remove preprocessed directories for a dataset (and optional subtask) to re-run preprocessing cleanly.
- **Inputs:**
  - Args: `--task dataset[:subtask]` e.g., `bdd100k:detection`, `nuscenes`, `carla`.
- **Outputs:**
  - Deletes `datasets/<dataset>/preprocessed[/<subtask>]` if present.
- **Notes:** Destructive; double-check target before running.
- **Run:**
  ```bash
  python3 scripts/redo_preprocess.py
  ```

---

### `test_carla.py`
- **Purpose:** Quick connectivity test to a running CARLA server; prints available maps.
- **Inputs:**
  - Requires CARLA server on `localhost:2000` and Python API.
- **Outputs:**
  - Console output only; no files written.
- **Run:**
  ```bash
  python3 scripts/test_carla.py
  ```

---

### `run_carla_scripts.sh`
- **Purpose:** Orchestrate multiple CARLA data collection runs with logging and resume support.
- **Inputs:**
  - Args: `--resume-from RUN_NUMBER`.
  - Env: `CARLA_DATA_PATH` for output root.
- **Outputs:**
  - Logs under `logs/`, plus datasets written by invoked Python collectors.
- **Run:**
  ```bash
  bash scripts/run_carla_scripts.sh
  ```

---

### `start_carla.sh`
- **Purpose:** Start CARLA server offscreen (Xvfb), set data path, and export environment variables.
- **Inputs:**
  - Env: `CARLA_DIR` (install directory), `CARLA_DATA_PATH` (output root). Starts Xvfb on `:1`.
- **Outputs:**
  - Launches CARLA (foreground); prints status and hints for connectivity test.
- **Run:**
  ```bash
  bash scripts/start_carla.sh
  ```


