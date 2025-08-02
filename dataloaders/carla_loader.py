import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

CARLA_ROOT = "datasets/carla/preprocessed"
BATCH_SIZE = 32
NUM_WORKERS = 4

class CarlaDataset(Dataset):
    def __init__(self, split="train", root_dir=CARLA_ROOT, transform=None):
        """
        Args:
            split (str): "train" or "val"
            root_dir (str): Path to preprocessed Carla data (should have train/ and val/ subfolders)
            transform (callable, optional): Optional transform to apply to the image tensor
        """
        self.split = split
        self.root_dir = Path(root_dir) / split
        self.transform = transform

        # Find all .pt files in all runs (recursively)
        self.pt_files = sorted([f for f in self.root_dir.rglob("*.pt")])

        if not self.pt_files:
            raise RuntimeError(f"No .pt files found in {self.root_dir}")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]
        sample = torch.load(pt_path)

        # Optionally transform the image on-the-fly
        if self.transform is not None and "image" in sample:
            sample["image"] = self.transform(sample["image"])
        return sample

def get_carla_loader(split='train', batch_size=None, num_workers=None, shuffle=None, transform=None):
    """
    Returns a PyTorch DataLoader for the Carla dataset.

    Args:
        split (str): 'train' or 'val'
        batch_size (int): Batch size for the loader
        num_workers (int): Number of DataLoader workers
        shuffle (bool): Shuffle data? (default: True for train, False for val)
        transform (callable): Optional transform for images

    Returns:
        DataLoader
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_workers is None:
        num_workers = NUM_WORKERS
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = CarlaDataset(split=split, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        collate_fn=lambda x: x
    )

def get_train_loader(batch_size=None, transform=None):
    return get_carla_loader('train', batch_size=batch_size, transform=transform)

def get_val_loader(batch_size=None, transform=None):
    return get_carla_loader('val', batch_size=batch_size, transform=transform)