import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from pathlib import Path

# Config
BASE_DIR = 'datasets/bdd100k/preprocessed/detection'
BATCH_SIZE = 32
NUM_WORKERS = 4

class BDD100KDetectionDataset(Dataset):
    def __init__(self, pt_dir, transform=None):
        self.pt_files = sorted([
            os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if f.endswith(".pt")
        ])
        self.transform = transform

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        sample = torch.load(self.pt_files[idx])
        image = read_image(sample["image_path"]).float() / 255.0
        bboxes = sample["bboxes"]
        labels = sample["labels"]
        meta = sample["meta"]

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "bboxes": bboxes,
            "labels": labels,
            "meta": meta
        }

def get_bdd_detection_loader(split='train', batch_size=None, num_workers=None, 
                           shuffle=None, transform=None):
    """
    Factory function to create BDD100K detection data loaders.
    
    Args:
        split: 'train', 'val', or 'test'
        batch_size: batch size (default: 32 for train, 1 for val/test)
        num_workers: number of workers (default: 4 for train, 1 for val/test)
        shuffle: whether to shuffle (default: True for train, False for val/test)
        transform: image transformations
    """
    # Set defaults based on split
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_workers is None:
        num_workers = NUM_WORKERS
    if shuffle is None:
        shuffle = (split == 'train')
    
    # Build path
    split_dir = Path(BASE_DIR) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    dataset = BDD100KDetectionDataset(split_dir, transform=transform)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),  # Only drop last for training
        collate_fn=lambda x: x
    )

# Convenience functions
def get_train_loader(batch_size=None, transform=None):
    return get_bdd_detection_loader('train', batch_size=batch_size, transform=transform)

def get_val_loader(batch_size=None, transform=None):
    return get_bdd_detection_loader('val', batch_size=batch_size, transform=transform)