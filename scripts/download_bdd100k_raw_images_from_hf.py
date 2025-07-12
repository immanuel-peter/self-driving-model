import os
from datasets import load_dataset
from tqdm import tqdm

def save_images(split: str, output_dir: str):
    dataset = load_dataset("immanuelpeter/bdd100k-raw-images", split=split)
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    # Try to get the length if possible, otherwise set to "unknown"
    num_images = getattr(dataset, "__len__", lambda: "unknown")()
    print(f"Saving {num_images} {split} images to {split_dir}")

    for sample in tqdm(dataset, desc=f"Downloading {split}"):
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
