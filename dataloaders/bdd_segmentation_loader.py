from pathlib import Path
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode

# Config
BDD100K_RAW_ROOT = 'datasets/bdd100k/raw'
BASE_DIR = 'datasets/bdd100k/preprocessed/segmentation'
BATCH_SIZE = 32
NUM_WORKERS = 4

class BDD100KSegmentationDataset(Dataset):
    def __init__(self, pt_dir, transform=None):
        self.pt_files = sorted(Path(pt_dir).glob("*.pt"))
        self.transform = transform
        self.raw_root = BDD100K_RAW_ROOT

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        sample = torch.load(self.pt_files[idx])

        def resolve_path(p):
            p = str(p)
            if os.path.isabs(p) and Path(p).exists():
                return p
            if Path(p).exists():
                return p
            if self.raw_root is not None:
                candidate = str(Path(self.raw_root) / p)
                if Path(candidate).exists():
                    return candidate
                if 'images' in p:
                    suffix = p.split('images', 1)[1]
                    candidate2 = str(Path(self.raw_root) / 'images' / suffix)
                    if Path(candidate2).exists():
                        return candidate2
            return p

        image_path = resolve_path(sample["image_path"])
        mask_path = resolve_path(sample["mask_path"])

        image = read_image(image_path).float() / 255.0
        mask = read_image(mask_path, mode=ImageReadMode.GRAY).squeeze(0).long()  # shape: [H, W]

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "mask": mask
        }

def get_bdd_segmentation_loader(split='train', batch_size=None, num_workers=None, 
                               shuffle=None, transform=None):
    """
    Factory function to create BDD100K segmentation data loaders.
    
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
    
    dataset = BDD100KSegmentationDataset(split_dir, transform=transform)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')  # Only drop last for training
    )