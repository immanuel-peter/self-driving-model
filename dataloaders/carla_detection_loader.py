from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_CARLA_PREPROCESSED_ROOT = "datasets/carla/preprocessed"
BATCH_SIZE = 32
NUM_WORKERS = 4

def _list_all_pt_files(split_dir: Path) -> List[Path]:\
    return sorted([p for p in split_dir.rglob("*.pt")])

def detection_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([b["image"] for b in batch], dim=0)

    max_boxes = 0
    for b in batch:
        max_boxes = max(max_boxes, int(b["bboxes"].shape[0]))

    if max_boxes == 0:
        padded_bboxes = torch.zeros((len(batch), 0, 4), dtype=torch.float32)
        padded_labels = torch.zeros((len(batch), 0), dtype=torch.int64)
    else:
        padded_bboxes = torch.full((len(batch), max_boxes, 4), -1.0, dtype=torch.float32)
        padded_labels = torch.full((len(batch), max_boxes), -1, dtype=torch.int64)
        for i, b in enumerate(batch):
            n = int(b["bboxes"].shape[0])
            if n > 0:
                padded_bboxes[i, :n] = b["bboxes"]
                padded_labels[i, :n] = b["labels"]

    return {
        "image": images,
        "bboxes": padded_bboxes,
        "labels": padded_labels,
    }


class CarlaDetectionDataset(Dataset):
    def __init__(self, split: str = "train", root_dir: str = DEFAULT_CARLA_PREPROCESSED_ROOT):
        self.root = Path(root_dir) / split
        if not self.root.exists():
            raise FileNotFoundError(f"Split directory not found: {self.root}")
        self.files = _list_all_pt_files(self.root)
        if not self.files:
            raise RuntimeError(f"No .pt files found under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = torch.load(self.files[idx], weights_only=False)

        image: torch.Tensor = sample["image"]  # [3, 256, 256]
        bboxes = sample.get("bboxes", None)
        labels = sample.get("labels", None)

        if bboxes is None or labels is None:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return {
            "image": image,
            "bboxes": bboxes,
            "labels": labels,
        }


def get_carla_detection_loader(
    split: str = "train",
    root_dir: str = DEFAULT_CARLA_PREPROCESSED_ROOT,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    shuffle: bool | None = None,
) -> DataLoader:
    if shuffle is None:
        shuffle = (split == "train")
    dataset = CarlaDetectionDataset(split=split, root_dir=root_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        collate_fn=detection_collate_fn,
    )


