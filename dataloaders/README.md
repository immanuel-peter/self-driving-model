## Dataloaders (Overview)

All dataloaders here consume preprocessed `.pt` files and return PyTorch `DataLoader`s with dictionaries of tensors.

For arguments, defaults, and environment variables, open each loader file.

> Note: Feel free to modify these files to align with your development workflow.

---

### `bdd_detection_loader.py`

- **Purpose**: BDD100K object detection.
- **Input**: `.pt` files under `datasets/bdd100k/preprocessed/detection/<split>/` with keys: `image_path`, `bboxes` `[N,4]`, `labels` `[N]`.
- **Output sample**: `{ image, bboxes, labels }` where boxes/labels are padded in the collate.
- **API**:
  - `BDD100KDetectionDataset(pt_dir, transform=None)`
  - `get_bdd_detection_loader(split='train', batch_size=None, num_workers=None, shuffle=None, transform=None)`
- **Notes**: Uses `detection_collate_fn` that pads to max boxes in batch with `-1`.

---

### `bdd_drivable_loader.py`

- **Purpose**: BDD100K drivable area segmentation (binary/ternary mask).
- **Input**: `.pt` under `datasets/bdd100k/preprocessed/drivable/<split>/` (override with `BDD100K_DRIVABLE_DIR`). Keys: `image_path`, `mask_path`.
- **Output sample**: `{ image, mask }` where `mask` is `[H,W]` long.
- **API**:
  - `BDD100KDrivableDataset(pt_dir, transform=None)`
  - `get_bdd_drivable_loader(split='train', ..., base_dir: str | None = None)`
- **Notes**: Robust channel handling for masks. Defaults: `BATCH_SIZE=32`, `NUM_WORKERS=4`.

---

### `bdd_segmentation_loader.py`

- **Purpose**: BDD100K semantic segmentation.
- **Input**: `.pt` under `datasets/bdd100k/preprocessed/segmentation/<split>/`. Each sample includes `image_path`, `mask_path`. If paths are relative, they are resolved via `BDD100K_RAW_ROOT`.
- **Output sample**: `{ image, mask }` where `mask` is `[H,W]` long (grayscale read).
- **API**:
  - `BDD100KSegmentationDataset(pt_dir, transform=None)`
  - `get_bdd_segmentation_loader(split='train', ...)`
- **Notes**: Path resolution logic handles absolute/relative/raw-root variants.

---

### `carla_detection_loader.py`

- **Purpose**: CARLA detection on preprocessed simulator frames.
- **Input**: `.pt` recursively under `datasets/carla/preprocessed/<split>/run_*/`. Keys: `image` `[3,H,W]` (already normalized), optional `bboxes` `[N,4]`, `labels` `[N]`.
- **Output sample**: `{ image, bboxes, labels }` (empty tensors if missing).
- **API**:
  - `CarlaDetectionDataset(split='train', root_dir=DEFAULT_CARLA_PREPROCESSED_ROOT)`
  - `get_carla_detection_loader(split='train', root_dir=..., batch_size=..., num_workers=..., shuffle=None)`
- **Notes**: Custom collate pads variable number of boxes/labels per batch.

---

### `carla_drivable_loader.py`

- **Purpose**: CARLA drivable area mask derived from CARLA semantic IDs.
- **Input**: `.pt` under `datasets/carla/preprocessed/<split>/`. Keys: `image`, `mask` (semantic IDs) may be present; if absent, returns ignore mask.
- **Output sample**: `{ image, mask }` where `mask` is mapped to `{0: background, 1: drivable, 2: alternative}`.
- **API**:
  - `CarlaDrivableDataset(split='train', root_dir=..., drivable_ids=None, alternative_ids=None)`
  - `get_carla_drivable_loader(split='train', root_dir=..., batch_size=..., num_workers=..., shuffle=None, drivable_ids=None, alternative_ids=None)`
- **Notes**: Configure IDs via args or env vars `CARLA_DRIVABLE_IDS`, `CARLA_ALTERNATIVE_IDS` (comma-separated).

---

### `carla_segmentation_loader.py`

- **Purpose**: Generic CARLA semantic segmentation.
- **Input**: `.pt` under `datasets/carla/preprocessed/<split>/` with keys: `image`, optional `mask`.
- **Output sample**: `{ image, mask }` where `mask` is `[H,W]` long; if missing, filled with ignore index `255`.
- **API**:
  - `CarlaSegmentationDataset(split='train', root_dir=...)`
  - `get_carla_segmentation_loader(split='train', root_dir=..., batch_size=..., num_workers=..., shuffle=None)`

---

### `carla_sequence_loader.py`

- **Purpose**: Build temporal windows for policy/trajectory learning from CARLA runs.
- **Input**: `.pt` under `datasets/carla/preprocessed/<split>/run_*/` containing keys like `image`, `vehicle_state` (location, rotation, speed_kmh, control), and optional `context`.
- **Output sample**: `{ image, waypoints [H,2], speed [H], throttle [H], steering [H], brake [H], context?, meta }` where waypoints are in ego-frame forward-right convention.
- **API**:
  - `CarlaSequenceDataset(split='train', root_dir=..., past=0, horizon=8, stride=1, include_context=True)`
  - `get_carla_sequence_loader(split='train', root_dir=..., batch_size=32, num_workers=4, past=0, horizon=8, stride=1, include_context=True, shuffle=None)`
- **Notes**: Converts world XY to ego XY using yaw; pre-indexes windows `[t .. t+horizon]` per run.

---

### `nuscenes_loader.py`

- **Purpose**: nuScenes 2D+LiDAR hybrid detection cache loader.
- **Input**: `.pt` under `datasets/nuscenes/preprocessed/<split>/` with keys like `image`, `lidar` `[N,3]`, `intrinsics`, and `boxes` (list of `Box` objects), `token`.
- **Output sample**: `{ image, lidar [B,max,3], intrinsics, boxes [B,max,7], labels [B,max], token }` with yaw extracted and classes mapped to 10 categories.
- **API**:
  - `NuScenesDataset(cache_dir)`
  - `get_nuscenes_loader(split='train', batch_size=None, num_workers=None, shuffle=None)`
- **Notes**: Collate pads variable-length LiDAR and boxes; adds safe globals for `Box` deserialization.

---

### Minimal usage pattern

```python
from dataloaders.carla_detection_loader import get_carla_detection_loader

loader = get_carla_detection_loader(split='train')
batch = next(iter(loader))
print(batch.keys())  # e.g., dict_keys(['image', 'bboxes', 'labels'])
```


