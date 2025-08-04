import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Default Config
CACHE_ROOT = 'datasets/nuscenes/preprocessed'
BATCH_SIZE = 32
NUM_WORKERS = 4
NUSCENES_CLASSES = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "trailer": 3,
    "construction_vehicle": 4,
    "pedestrian": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic_cone": 8,
    "barrier": 9,
}

def nuscenes_collate_fn(batch):
    """
    Custom collate function for the NuScenes dataset.

    It handles the variable number of LiDAR points and ground-truth boxes
    by padding them to the maximum length in the batch.

    Args:
        batch: A list of sample dictionaries from the NuScenesDataset.

    Returns:
        A single dictionary containing the batched and padded tensors.
    """
    # Fixed-size tensors can be stacked directly
    images = torch.stack([item['image'] for item in batch], dim=0)
    intrinsics = torch.stack([item['intrinsics'] for item in batch], dim=0)

    # --- Pad LiDAR point clouds ---
    max_lidar_points = max(item['lidar'].shape[0] for item in batch)
    padded_lidars = torch.zeros(
        (len(batch), max_lidar_points, 3),
        dtype=torch.float32
    )
    for i, item in enumerate(batch):
        pts = item['lidar']
        padded_lidars[i, :pts.shape[0], :] = pts

    # --- Convert Box objects to tensor + labels ---
    def box_list_to_tensor_and_labels(box_list):
        """
        Turn a list of NuScenes Box objects into:
          - boxes:  (N,7) tensor [cx, cy, cz, w, l, h, yaw]
          - labels: (N,) tensor of class IDs in [0..9]
        """
        if not box_list:
            return torch.zeros((0,7),dtype=torch.float32), torch.zeros((0,),dtype=torch.int64)
        feats = []
        labels = []
        for b in box_list:
            coords = list(b.center) + list(b.wlh) + [b.orientation.yaw]
            feats.append(torch.tensor(coords, dtype=torch.float32))
            # map b.name -> integer class
            labels.append(NUSCENES_CLASSES[b.name])
        return torch.stack(feats, dim=0), torch.tensor(labels, dtype=torch.int64)

    for item in batch:
        boxes_t, labels_t = box_list_to_tensor_and_labels(item['boxes'])
        item['boxes'] = boxes_t
        item['labels'] = labels_t

    # --- Pad Ground-Truth Boxes & Labels ---
    max_boxes = max(item['boxes'].shape[0] for item in batch)
    padded_boxes = torch.full(
        (len(batch), max_boxes, 7),
        fill_value=-1.0, dtype=torch.float32
    )
    padded_labels = torch.full(
        (len(batch), max_boxes),
        fill_value=-1, dtype=torch.int64
    )
    for i, item in enumerate(batch):
        n = item['boxes'].shape[0]
        if n > 0:
            padded_boxes[i, :n, :] = item['boxes']
            padded_labels[i, :n]   = item['labels']

    # --- Collect tokens ---
    tokens = [item['token'] for item in batch]

    return {
        'image':      images,
        'lidar':      padded_lidars,
        'intrinsics': intrinsics,
        'boxes':      padded_boxes,   # [B, max_boxes, 7]
        'labels':     padded_labels,  # [B, max_boxes]
        'token':      tokens
    }

class NuScenesDataset(Dataset):
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.files = sorted([
            os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])

def get_nuscenes_loader(split='train', batch_size=None, num_workers=None, 
                         shuffle=None):
    """
    Factory function to create NuScenes data loaders from cached .pt files.

    Args:
        split: 'train' or 'val'
        batch_size: batch size (default: 32 for train, 1 for val)
        num_workers: number of workers (default: 4 for train, 1 for val)
        shuffle: whether to shuffle (default: True for train, False for val)
        cache_root: root directory where split folders are stored
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_workers is None:
        num_workers = NUM_WORKERS
    if shuffle is None:
        shuffle = (split == 'train')

    split_dir = Path(CACHE_ROOT) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    dataset = NuScenesDataset(split_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        collate_fn=nuscenes_collate_fn
    )
