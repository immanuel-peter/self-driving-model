import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

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

def get_bdd_detection_loader(split_dir, batch_size=32, shuffle=True, transform=None):
    dataset = BDD100KDetectionDataset(split_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)
