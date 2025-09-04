from pathlib import Path
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

BASE_DIR = 'datasets/bdd100k/preprocessed/drivable'
BATCH_SIZE = 32
NUM_WORKERS = 4

class BDD100KDrivableDataset(Dataset):
    def __init__(self, pt_dir, transform=None):
        self.pt_files = sorted(Path(pt_dir).glob("*.pt"))
        self.transform = transform

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        sample = torch.load(self.pt_files[idx], weights_only=False)
        image = read_image(sample["image_path"]).float() / 255.0
        mask_t = read_image(sample["mask_path"]).long()  # [C,H,W]
        if mask_t.dim() == 3 and mask_t.shape[0] > 1:
            mask = mask_t[0]
        else:
            mask = mask_t.squeeze(0)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "mask": mask
        }

def get_bdd_drivable_loader(split='train', batch_size=None, num_workers=None, 
                           shuffle=None, transform=None, base_dir: str | None = None):
    """
    Factory function to create BDD100K drivable area data loaders.
    
    Args:
        split: 'train', 'val', or 'test'
        batch_size: batch size (default: 32 for train, 1 for val/test)
        num_workers: number of workers (default: 4 for train, 1 for val/test)
        shuffle: whether to shuffle (default: True for train, False for val/test)
        transform: image transformations
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_workers is None:
        num_workers = NUM_WORKERS
    if shuffle is None:
        shuffle = (split == 'train')
    
    root = Path(base_dir) if base_dir is not None else Path(BASE_DIR)
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    dataset = BDD100KDrivableDataset(split_dir, transform=transform)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )