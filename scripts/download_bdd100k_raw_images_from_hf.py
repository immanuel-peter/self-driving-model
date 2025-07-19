import os
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from tqdm import tqdm

def get_dataset_length(dataset):
    if isinstance(dataset, (Dataset, DatasetDict)):
        return len(dataset)
    elif isinstance(dataset, IterableDataset):
        # IterableDataset doesn't support len()
        print("Cannot get length of IterableDataset - it's a streaming dataset")
        return None
    elif isinstance(dataset, IterableDatasetDict):
        # Returns number of splits, not total examples
        return len(dataset)  # This gives you the number of splits
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

def save_images(split: str, output_dir: str):
    dataset = load_dataset("immanuelpeter/bdd100k-raw-images", split=split)
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    # Try to get the length if possible, otherwise set to "unknown"
    num_images = get_dataset_length(dataset)
    print(f"Saving {num_images} {split} images to {split_dir}")

    for sample in tqdm(dataset, desc=f"Downloading {split}", total=num_images):
        image = sample["image"]
        filename = sample["filename"] + ".jpg"  # assuming .jpg, adapt if needed
        save_path = os.path.join(split_dir, filename)

        # Save image (will be in RGB or grayscale depending on original)
        img = image.convert("RGB")
        img.save(save_path)

def main():
    target_root = "datasets/bdd100k/raw/images/100k"
    splits = ["train", "val", "test"]

    for split in splits:
        save_images(split, target_root)

if __name__ == "__main__":
    main()
