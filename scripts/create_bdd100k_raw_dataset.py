"""
Script to create and push a Hugging Face dataset for BDD100K raw images.
"""

import os
from pathlib import Path
from typing import Union
from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Hugging Face username
username = os.getenv("HF_USERNAME") or "immanuelpeter"
if not username:
    raise ValueError("HF_USERNAME environment variable is not set")

def get_image_files(split_dir: Union[str, Path]) -> list[str]:
    """Get all image files from a split directory."""
    split_dir = Path(split_dir)
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []

    for ext in image_extensions:
        image_files.extend(split_dir.glob(ext))
        image_files.extend(split_dir.glob(ext.upper()))

    return sorted([str(p.resolve()) for p in image_files])

def load_image(image_path: str) -> PILImage.Image | None:
    """Verify image can be opened; return None if not."""
    try:
        with PILImage.open(image_path) as img:
            img.verify()
        return img
    except Exception as e:
        logger.warning(f"Error loading image {image_path}: {e}")
        return None

def create_dataset_from_images(image_dir: Path, split_name: str) -> list[dict]:
    """Build list of image metadata for a dataset split."""
    logger.info(f"Processing {split_name} split...")

    image_files = get_image_files(image_dir)
    logger.info(f"Found {len(image_files)} images in {split_name}")

    dataset_data = []

    for image_path in tqdm(image_files, desc=f"Loading {split_name} images"):
        if not load_image(image_path):
            continue

        filename = os.path.splitext(os.path.basename(image_path))[0]
        dataset_data.append({
            "image": image_path,
            "filename": filename,
            "split": split_name,
        })

    logger.info(f"Split {split_name} ready with {len(dataset_data)} valid images")
    return dataset_data

def main():
    base_path = Path("datasets/bdd100k/raw/images/100k").resolve()

    if not base_path.exists():
        logger.error(f"Base path {base_path} does not exist!")
        return

    splits = {
        "train": base_path / "train",
        "val": base_path / "val",
        "test": base_path / "test"
    }

    all_datasets = {}

    for split_name, split_dir in splits.items():
        if not split_dir.exists():
            logger.warning(f"Split path {split_dir} missing, skipping...")
            continue

        data = create_dataset_from_images(split_dir, split_name)
        if data:
            all_datasets[split_name] = data

    if not all_datasets:
        logger.error("No data collected — aborting upload.")
        return

    features = Features({
        "image": Image(),
        "filename": Value("string"),
        "split": Value("string"),
    })

    dataset_dict = DatasetDict({
        split: Dataset.from_list(data, features=features)
        for split, data in all_datasets.items()
    })

    repo_id = f"{username}/bdd100k-raw-images"
    logger.info(f"Pushing to Hugging Face Hub: {repo_id}")

    try:
        dataset_dict.push_to_hub(repo_id)
        logger.info(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        logger.error(f"Failed to push to Hub: {e}")
        return

    # Log summary stats
    for split_name in dataset_dict:
        logger.info(f"{split_name}: {len(dataset_dict[split_name])} samples")

if __name__ == "__main__":
    main()
