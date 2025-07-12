from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

class BDD100KSegmentationDataset(Dataset):
    def __init__(self, pt_dir, transform=None):
        self.pt_files = sorted(Path(pt_dir).glob("*.pt"))
        self.transform = transform

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        sample = torch.load(self.pt_files[idx])
        image = read_image(sample["image_path"]).float() / 255.0
        mask = read_image(sample["mask_path"]).squeeze(0).long()  # shape: [H, W]

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "mask": mask
        }

def get_bdd_segmentation_loader(split_dir, batch_size=32, shuffle=True, transform=None):
    dataset = BDD100KSegmentationDataset(split_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 