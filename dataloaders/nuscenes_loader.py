import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Default Config
CACHE_ROOT = 'datasets/nuscenes/preprocessed'
BATCH_SIZE = 32
NUM_WORKERS = 4

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
                         shuffle=None, cache_root=None):
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
    if cache_root is None:
        cache_root = CACHE_ROOT

    split_dir = Path(cache_root) / split
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
        collate_fn=lambda x: x
    )

# Convenience accessors
def get_train_loader(batch_size=None):
    return get_nuscenes_loader('train', batch_size=batch_size)

def get_val_loader(batch_size=None):
    return get_nuscenes_loader('val', batch_size=batch_size)
