#!/usr/bin/env python3
"""
Script to create a Hugging Face dataset from BDD100K raw images.
"""

import os
import glob
from pathlib import Path
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_image_files(split_dir):
    """Get all image files from a split directory."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(split_dir, ext)))
        image_files.extend(glob.glob(os.path.join(split_dir, ext.upper())))
    
    return sorted(image_files)

def load_image(image_path):
    """Load and return a PIL Image."""
    try:
        return PILImage.open(image_path)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

def create_dataset_from_images(image_dir, split_name):
    """Create a dataset from images in a directory."""
    logger.info(f"Processing {split_name} split...")
    
    image_files = get_image_files(image_dir)
    logger.info(f"Found {len(image_files)} images in {split_name}")
    
    if not image_files:
        logger.warning(f"No images found in {image_dir}")
        return None
    
    # Prepare dataset data
    dataset_data = []
    
    for i, image_path in enumerate(image_files):
        if i % 1000 == 0:
            logger.info(f"Processing image {i+1}/{len(image_files)}")
        
        # Load image
        image = load_image(image_path)
        if image is None:
            continue
        
        # Get filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        dataset_data.append({
            'image': image,
            'filename': filename,
            'split': split_name,
            'file_path': image_path
        })
    
    logger.info(f"Created dataset with {len(dataset_data)} samples for {split_name}")
    return dataset_data

def main():
    """Main function to create the BDD100K raw images dataset."""
    # Define paths
    base_path = Path("datasets/bdd100k/raw/images/100k")
    
    # Check if base path exists
    if not base_path.exists():
        logger.error(f"Base path {base_path} does not exist!")
        return
    
    # Define splits
    splits = {
        'train': base_path / 'train',
        'test': base_path / 'test', 
        'val': base_path / 'val'
    }
    
    # Create datasets for each split
    all_datasets = {}
    
    for split_name, split_path in splits.items():
        if not split_path.exists():
            logger.warning(f"Split path {split_path} does not exist, skipping...")
            continue
            
        dataset_data = create_dataset_from_images(split_path, split_name)
        if dataset_data:
            all_datasets[split_name] = dataset_data
    
    # Combine all datasets
    if not all_datasets:
        logger.error("No datasets created!")
        return
    
    # Create the main dataset
    all_data = []
    for split_name, data in all_datasets.items():
        all_data.extend(data)
    
    logger.info(f"Creating dataset with {len(all_data)} total samples...")
    
    # Define features
    features = Features({
        'image': Image(),
        'filename': Value('string'),
        'split': Value('string'),
        'file_path': Value('string')
    })
    
    # Create dataset
    dataset = Dataset.from_list(all_data, features=features)
    
    # Save dataset
    output_dir = "datasets/bdd100k/hf-raw-images"
    dataset.save_to_disk(output_dir)
    
    logger.info(f"Dataset saved to {output_dir}")
    logger.info(f"Dataset info: {dataset}")
    
    # Print split statistics
    for split_name in splits.keys():
        split_data = dataset.filter(lambda x: x['split'] == split_name)
        logger.info(f"{split_name}: {len(split_data)} samples")

if __name__ == "__main__":
    main() 