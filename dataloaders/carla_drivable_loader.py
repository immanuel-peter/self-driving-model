from pathlib import Path
import os
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_CARLA_PREPROCESSED_ROOT = "datasets/carla/preprocessed"
BATCH_SIZE = 32
NUM_WORKERS = 4

def _parse_ids_env(env_key: str) -> Optional[List[int]]:
    val = os.environ.get(env_key)
    if not val:
        return None
    try:
        return [int(x.strip()) for x in val.split(',') if x.strip()]
    except Exception:
        return None

class CarlaDrivableDataset(Dataset):
    def __init__(self,
                 split: str = "train",
                 root_dir: str = DEFAULT_CARLA_PREPROCESSED_ROOT,
                 drivable_ids: Optional[List[int]] = None,
                 alternative_ids: Optional[List[int]] = None):
        self.root = Path(root_dir) / split
        if not self.root.exists():
            raise FileNotFoundError(f"Split directory not found: {self.root}")
        self.files = sorted(self.root.rglob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No .pt files found under {self.root}")

        env_drv = _parse_ids_env('CARLA_DRIVABLE_IDS')
        env_alt = _parse_ids_env('CARLA_ALTERNATIVE_IDS')
        self.drivable_ids = drivable_ids if drivable_ids is not None else (env_drv if env_drv is not None else [7])
        self.alternative_ids = alternative_ids if alternative_ids is not None else (env_alt if env_alt is not None else [])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = torch.load(self.files[idx], weights_only=False)
        image: torch.Tensor = sample["image"]
        raw_mask: torch.Tensor = sample.get("mask", None)
        if raw_mask is None:
            # All ignore if no mask
            H, W = int(image.shape[-2]), int(image.shape[-1])
            out_mask = torch.full((H, W), 255, dtype=torch.long)
        else:
            # Normalize incoming mask to 2D long tensor [H, W]
            if raw_mask.dim() == 3:
                if raw_mask.shape[-1] in (3, 4):
                    raw_mask = raw_mask[..., 0]
                elif raw_mask.shape[0] in (3, 4):
                    raw_mask = raw_mask[0, ...]
                else:
                    raw_mask = raw_mask.squeeze()
            raw_mask = raw_mask.to(dtype=torch.long)
            # Map CARLA semantic IDs to {0: background, 1: drivable, 2: alternative}
            # Start as background
            out_mask = torch.zeros_like(raw_mask, dtype=torch.long)
            # Drivable
            for cid in self.drivable_ids:
                out_mask[raw_mask == cid] = 1
            # Alternative
            for cid in self.alternative_ids:
                out_mask[raw_mask == cid] = 2

        return {
            "image": image,
            "mask": out_mask,
        }


def get_carla_drivable_loader(
    split: str = "train",
    root_dir: str = DEFAULT_CARLA_PREPROCESSED_ROOT,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    shuffle: bool | None = None,
    drivable_ids: Optional[List[int]] = None,
    alternative_ids: Optional[List[int]] = None,
) -> DataLoader:
    if shuffle is None:
        shuffle = (split == "train")
    dataset = CarlaDrivableDataset(
        split=split,
        root_dir=root_dir,
        drivable_ids=drivable_ids,
        alternative_ids=alternative_ids,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


