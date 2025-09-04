from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_CARLA_PREPROCESSED_ROOT = "datasets/carla/preprocessed"
BATCH_SIZE = 32
NUM_WORKERS = 4

class CarlaSegmentationDataset(Dataset):
    def __init__(self, split: str = "train", root_dir: str = DEFAULT_CARLA_PREPROCESSED_ROOT):
        self.root = Path(root_dir) / split
        if not self.root.exists():
            raise FileNotFoundError(f"Split directory not found: {self.root}")
        self.files = sorted(self.root.rglob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No .pt files found under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = torch.load(self.files[idx], weights_only=False)
        image: torch.Tensor = sample["image"]
        mask = sample.get("mask", None)
        if mask is None:
            mask = torch.full((image.shape[-2], image.shape[-1]), 255, dtype=torch.long)
        else:
            # Normalize mask to 2D long tensor [H, W]
            if mask.dim() == 3:
                # [H,W,C] or [C,H,W]
                if mask.shape[-1] in (3, 4):
                    mask = mask[..., 0]
                elif mask.shape[0] in (3, 4):
                    mask = mask[0, ...]
                else:
                    mask = mask.squeeze()
            mask = mask.to(dtype=torch.long)
        return {
            "image": image,
            "mask": mask,
        }


def get_carla_segmentation_loader(
    split: str = "train",
    root_dir: str = DEFAULT_CARLA_PREPROCESSED_ROOT,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    shuffle: bool | None = None,
) -> DataLoader:
    if shuffle is None:
        shuffle = (split == "train")
    dataset = CarlaSegmentationDataset(split=split, root_dir=root_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


