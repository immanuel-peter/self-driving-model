"""
Push CARLA autopilot run dataset to Hugging Face Datasets hub.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import logging
from tqdm import tqdm
import random

from datasets import Dataset, Features, Value, Image as HFImage, Sequence, DatasetDict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Hugging Face username
username = os.getenv("HF_USERNAME") or "immanuelpeter"
if not username:
    raise ValueError("HF_USERNAME environment variable is not set")

def parse_vehicle_log(log_path: Path) -> List[Dict[str, Any]]:
    try:
        with open(log_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse {log_path}: {e}")
        return []

def parse_config(config_path: Path) -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse {config_path}: {e}")
        return {}

def get_available_cameras(images_dir: Path) -> List[str]:
    cameras = []
    for cam_dir in images_dir.iterdir():
        if cam_dir.is_dir() and any(cam_dir.glob("*.png")):
            cameras.append(cam_dir.name)
    return sorted(cameras)

def collect_image_paths(images_dir: Path, frame_data: Dict[str, Any], cameras: List[str]) -> Dict[str, Any]:
    image_filename = frame_data["image_filename"]
    image_paths = {}
    for camera in cameras:
        camera_dir = images_dir / camera
        image_path = camera_dir / image_filename
        # Hugging Face expects None if missing
        image_paths[f"image_{camera}"] = str(image_path) if image_path.exists() else None
    return image_paths

def process_run(run_dir: Path, run_id: str) -> List[Dict[str, Any]]:
    logger.info(f"Processing {run_id}")

    config_path = run_dir / "config.json"
    config = parse_config(config_path)
    log_path = run_dir / "vehicle_log.json"
    vehicle_data = parse_vehicle_log(log_path)

    if not vehicle_data:
        logger.warning(f"No vehicle data found for {run_id}")
        return []

    images_dir = run_dir / "images"
    cameras = get_available_cameras(images_dir)
    logger.info(f"Found cameras for {run_id}: {cameras}")

    processed_data = []
    for frame_data in tqdm(vehicle_data, desc=f"Processing {run_id}"):
        image_paths = collect_image_paths(images_dir, frame_data, cameras)
        # Derived file stems and auxiliary modalities (front camera convention)
        image_filename = frame_data["image_filename"]
        stem = Path(image_filename).stem
        # Prefer colorized visualization for segmentation masks
        seg_front_vis = run_dir / "segmentation_vis" / "front" / image_filename
        seg_front_raw = run_dir / "segmentation" / "front" / image_filename
        seg_front_path = seg_front_vis if seg_front_vis.exists() else seg_front_raw
        lidar_path = run_dir / "lidar" / f"{stem}.npy"
        ann_path = run_dir / "annots" / "front" / f"{stem}.json"

        # Parse 2D boxes (if available)
        boxes = []
        box_labels = []
        if ann_path.exists():
            try:
                with open(ann_path, 'r') as f:
                    ann = json.load(f)
                for obj in ann.get('boxes', []):
                    bbox = obj.get('bbox')
                    label = obj.get('label', 'vehicle')
                    if bbox and len(bbox) == 4:
                        boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
                        box_labels.append(str(label))
            except Exception as e:
                logger.warning(f"Failed to parse {ann_path}: {e}")

        # Load LiDAR points into-memory (Nx4: x,y,z,intensity). Store as nested lists.
        lidar_points = []
        if lidar_path.exists():
            try:
                pts = np.load(lidar_path)
                if pts.ndim == 2 and pts.shape[1] >= 3:
                    # Ensure float32 lists for Arrow serialization
                    lidar_points = pts[:, :4] if pts.shape[1] >= 4 else np.pad(pts[:, :3], ((0,0),(0,1)), constant_values=0.0)
                    lidar_points = lidar_points.astype(np.float32).tolist()
            except Exception as e:
                logger.warning(f"Failed to load LiDAR {lidar_path}: {e}")

        sample = {
            "run_id": run_id,
            "frame": frame_data["frame"],
            "timestamp": frame_data["timestamp"],
            **image_paths,
            # Optional modalities
            "seg_front": str(seg_front_path) if seg_front_path.exists() else None,
            "lidar": lidar_points,
            "boxes": boxes,
            "box_labels": box_labels,
            "location_x": frame_data["location"]["x"],
            "location_y": frame_data["location"]["y"],
            "location_z": frame_data["location"]["z"],
            "rotation_pitch": frame_data["rotation"]["pitch"],
            "rotation_yaw": frame_data["rotation"]["yaw"],
            "rotation_roll": frame_data["rotation"]["roll"],
            "velocity_x": frame_data["velocity"]["x"],
            "velocity_y": frame_data["velocity"]["y"],
            "velocity_z": frame_data["velocity"]["z"],
            "speed_kmh": frame_data["speed_kmh"],
            "throttle": frame_data["control"]["throttle"],
            "steer": frame_data["control"]["steer"],
            "brake": frame_data["control"]["brake"],
            "nearby_vehicles_50m": frame_data["traffic_density"]["nearby_vehicles_50m"],
            "total_npc_vehicles": frame_data["traffic_density"]["total_npc_vehicles"],
            "total_npc_walkers": frame_data["traffic_density"]["total_npc_walkers"],
            "map_name": config.get("map", ""),
            "weather_cloudiness": config.get("weather", {}).get("cloudiness", 0.0),
            "weather_precipitation": config.get("weather", {}).get("precipitation", 0.0),
            "weather_fog_density": config.get("weather", {}).get("fog_density", 0.0),
            "weather_sun_altitude": config.get("weather", {}).get("sun_altitude_angle", 0.0),
            "vehicles_spawned": config.get("npc_config", {}).get("vehicles_spawned", 0),
            "walkers_spawned": config.get("npc_config", {}).get("walkers_spawned", 0),
            "duration_seconds": config.get("duration_seconds", 0),
        }
        processed_data.append(sample)
    return processed_data

def create_dataset_features(example_samples: List[Dict[str, Any]]) -> Features:
    if not example_samples:
        raise ValueError("No samples provided for feature creation")
    
    # Use the first sample to determine the features
    example_sample = example_samples[0]
    features = {
        "run_id": Value("string"),
        "frame": Value("int32"),
        "timestamp": Value("float32"),
        "location_x": Value("float32"),
        "location_y": Value("float32"),
        "location_z": Value("float32"),
        "rotation_pitch": Value("float32"),
        "rotation_yaw": Value("float32"),
        "rotation_roll": Value("float32"),
        "velocity_x": Value("float32"),
        "velocity_y": Value("float32"),
        "velocity_z": Value("float32"),
        "speed_kmh": Value("float32"),
        "throttle": Value("float32"),
        "steer": Value("float32"),
        "brake": Value("float32"),
        "nearby_vehicles_50m": Value("int32"),
        "total_npc_vehicles": Value("int32"),
        "total_npc_walkers": Value("int32"),
        "map_name": Value("string"),
        "weather_cloudiness": Value("float32"),
        "weather_precipitation": Value("float32"),
        "weather_fog_density": Value("float32"),
        "weather_sun_altitude": Value("float32"),
        "vehicles_spawned": Value("int32"),
        "walkers_spawned": Value("int32"),
        "duration_seconds": Value("int32"),
        # Optional modalities
        "seg_front": HFImage(),
        "lidar": Sequence(Sequence(Value("float32"))),
        "boxes": Sequence(Sequence(Value("float32"))),
        "box_labels": Sequence(Value("string")),
    }
    
    # Add image fields dynamically
    for key in example_sample:
        if key.startswith("image_"):
            features[key] = HFImage()
    
    return Features(features)

def main():
    parser = argparse.ArgumentParser(description="Convert CARLA dataset to HuggingFace format")
    parser.add_argument("--raw_dir", type=str, default="datasets/carla/raw",
                       help="Path to raw CARLA dataset directory")
    parser.add_argument("--dataset_name", type=str, default="carla-autopilot-images-v2",
                       help="Name for the HuggingFace dataset")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push dataset to HuggingFace Hub")
    parser.add_argument("--max_runs", type=int, default=None,
                       help="Maximum number of runs to process (for testing)")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Fraction of runs to use for training")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Fraction of runs to use for validation")
    parser.add_argument("--test_split", type=float, default=0.1,
                       help="Fraction of runs to use for test")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")
    # Validate splits
    total_split = args.train_split + args.val_split + args.test_split
    if not abs(total_split - 1.0) < 1e-6:
        raise ValueError(f"Splits must sum to 1.0 (got {total_split})")
    # Find all run directories
    run_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if args.max_runs:
        run_dirs = run_dirs[:args.max_runs]
    # Shuffle and split runs for reproducibility
    random.seed(args.seed)
    random.shuffle(run_dirs)
    n_total = len(run_dirs)
    n_train = int(n_total * args.train_split)
    n_val = int(n_total * args.val_split)
    n_test = n_total - n_train - n_val
    train_run_dirs = run_dirs[:n_train]
    val_run_dirs = run_dirs[n_train:n_train+n_val]
    test_run_dirs = run_dirs[n_train+n_val:]
    logger.info(f"Found {n_total} runs to process: {len(train_run_dirs)} train, {len(val_run_dirs)} val, {len(test_run_dirs)} test")
    # Process all runs for each split
    train_data = []
    for run_dir in tqdm(train_run_dirs, desc="Processing train runs"):
        train_data.extend(process_run(run_dir, run_dir.name))
    val_data = []
    for run_dir in tqdm(val_run_dirs, desc="Processing val runs"):
        val_data.extend(process_run(run_dir, run_dir.name))
    test_data = []
    for run_dir in tqdm(test_run_dirs, desc="Processing test runs"):
        test_data.extend(process_run(run_dir, run_dir.name))
    if not train_data or not val_data or not test_data:
        raise ValueError("No data was processed successfully for one or more splits")
    logger.info(f"Processed {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
    # Create dataset features
    features = create_dataset_features(train_data + val_data + test_data)
    # Create datasets
    logger.info("Creating HuggingFace datasets...")
    train_dataset = Dataset.from_list(train_data, features=features)
    val_dataset = Dataset.from_list(val_data, features=features)
    test_dataset = Dataset.from_list(test_data, features=features)
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    repo_id = f"{username}/{args.dataset_name}"
    # Push to hub if requested
    if args.push_to_hub:
        logger.info(f"Pushing dataset to HuggingFace Hub as {repo_id}")
        dataset_dict.push_to_hub(repo_id, private=False)
        logger.info("Dataset creation completed and uploaded to the Hub!")
        logger.info(f"Train samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        logger.info(f"Test samples: {len(test_data)}")
    else:
        logger.warning("--push_to_hub not set. The dataset will not be saved or uploaded anywhere.")

if __name__ == "__main__":
    main()