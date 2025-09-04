import json
import torch
import argparse
import math
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torchvision import transforms

RAW_DIR = "datasets/carla/raw"
PREPROCESSED_DIR = "datasets/carla/preprocessed"
CAMERA_CONFIGS = ["front"]

def get_transforms():
    """Get image preprocessing transforms similar to other datasets"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_vehicle_log(log_path):
    """Load and index vehicle log by frame number"""
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    frame_lookup = {}
    for entry in log_data:
        frame_lookup[entry['frame']] = entry
    
    return frame_lookup

def load_config(config_path):
    """Load run configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)

def parse_weather_info(weather_config):
    """Extract weather features for the model"""
    return {
        'cloudiness': weather_config.get('cloudiness', 0.0),
        'precipitation': weather_config.get('precipitation', 0.0),
        'wetness': weather_config.get('wetness', 0.0),
        'fog_density': weather_config.get('fog_density', 0.0),
        'sun_altitude_angle': weather_config.get('sun_altitude_angle', 0.0)
    }

def process_frame(run_dir, frame_id, frame_data, config, transform):
    """Process a single frame and return the data structure"""
    
    images = {}
    image_filename = frame_data['image_filename']
    stem = Path(image_filename).stem
    
    for cam_name in CAMERA_CONFIGS:
        img_path = run_dir / "images" / cam_name / image_filename
        if img_path.exists():
            try:
                image = Image.open(img_path).convert('RGB')
                images[cam_name] = transform(image)
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
                continue
        else:
            print(f"Warning: Missing image {img_path}")
    
    if 'front' not in images:
        return None
    
    location = frame_data['location']
    rotation = frame_data['rotation'] 
    velocity = frame_data['velocity']
    control = frame_data['control']
    traffic = frame_data['traffic_density']
    
    weather_features = parse_weather_info(config['weather'])
    
    mask_tensor = None
    seg_path = run_dir / "segmentation" / "front" / image_filename
    if seg_path.exists():
        try:
            mask_img = Image.open(seg_path)
            mask_img = mask_img.resize((256, 256), resample=Image.NEAREST)
            mask_np = np.array(mask_img).astype(np.int64)
            mask_tensor = torch.from_numpy(mask_np)
        except Exception as e:
            print(f"Warning: Failed to load segmentation {seg_path}: {e}")
            mask_tensor = None
    
    bboxes_tensor = None
    labels_tensor = None
    ann_path = run_dir / "annots" / "front" / f"{stem}.json"
    if ann_path.exists():
        try:
            with open(ann_path, 'r') as f:
                ann = json.load(f)
            boxes = []
            labels = []
            cls_map = {
                'vehicle': 0,
                'pedestrian': 1,
            }
            raw_w, raw_h = 800, 600
            sx, sy = 256.0 / raw_w, 256.0 / raw_h
            for obj in ann.get('boxes', []):
                bbox = obj.get('bbox')
                label_str = obj.get('label', 'vehicle')
                if not bbox or label_str not in cls_map:
                    continue
                x1, y1, x2, y2 = bbox
                boxes.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
                labels.append(cls_map[label_str])
            if len(boxes) > 0:
                bboxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.int64)
            else:
                bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                labels_tensor = torch.zeros((0,), dtype=torch.int64)
        except Exception as e:
            print(f"Warning: Failed to load annots {ann_path}: {e}")
            bboxes_tensor = None
            labels_tensor = None
    
    lidar_tensor = None
    lidar_path = run_dir / "lidar" / f"{stem}.npy"
    if lidar_path.exists():
        try:
            pts = np.load(lidar_path)  # Nx4 (x,y,z,intensity)
            if pts.ndim == 2 and pts.shape[1] >= 3:
                lidar_tensor = torch.from_numpy(pts[:, :3].astype(np.float32))
        except Exception as e:
            print(f"Warning: Failed to load LiDAR {lidar_path}: {e}")
            lidar_tensor = None
    
    def build_camera_intrinsic(width, height, fov_deg):
        f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        K = torch.tensor([
            [f, 0.0, width / 2.0],
            [0.0, f, height / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        return K
    K_raw = build_camera_intrinsic(800, 600, 90)
    sx, sy = 256.0 / 800.0, 256.0 / 600.0
    S = torch.tensor([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    K_resized = S @ K_raw
    
    sample = {
        'image': images['front'],
        'mask': mask_tensor if mask_tensor is not None else None,
        'bboxes': bboxes_tensor if bboxes_tensor is not None else None,
        'labels': labels_tensor if labels_tensor is not None else None,
        'lidar': lidar_tensor if lidar_tensor is not None else None,
        'intrinsics': K_resized,
        
        # Vehicle state (similar to control data in other datasets)
        'vehicle_state': {
            'location': torch.tensor([location['x'], location['y'], location['z']], dtype=torch.float32),
            'rotation': torch.tensor([rotation['pitch'], rotation['yaw'], rotation['roll']], dtype=torch.float32),
            'velocity': torch.tensor([velocity['x'], velocity['y'], velocity['z']], dtype=torch.float32),
            'speed_kmh': torch.tensor(frame_data['speed_kmh'], dtype=torch.float32),
            'control': torch.tensor([control['throttle'], control['steer'], control['brake']], dtype=torch.float32)
        },
        
        # Environmental context (for expert selection/routing)
        'context': {
            'weather': torch.tensor([
                weather_features['cloudiness'] / 100.0,
                weather_features['precipitation'] / 100.0,
                weather_features['wetness'] / 100.0,
                weather_features['fog_density'] / 100.0,
                (weather_features['sun_altitude_angle'] + 90) / 180.0  # Normalize to [0,1]
            ], dtype=torch.float32),
            'traffic_density': torch.tensor([
                traffic['nearby_vehicles_50m'],
                traffic['total_npc_vehicles'], 
                traffic['total_npc_walkers']
            ], dtype=torch.float32)
        },
        
        # Metadata
        'meta': {
            'frame_id': frame_data['frame'],
            'timestamp': frame_data['timestamp'],
            'run_id': config['run_id'],
            'map': config['map'],
            'camera': 'front',
            'image_path': str(run_dir / 'images' / 'front' / image_filename),
            'seg_path': str(seg_path) if seg_path.exists() else '',
            'lidar_path': str(lidar_path) if lidar_path.exists() else '',
            'ann_path': str(ann_path) if ann_path.exists() else ''
        }
    }
    
    return sample

def process_run(run_dir, output_dir, transform):
    """Process a single CARLA run"""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    
    config_path = run_dir / "config.json"
    log_path = run_dir / "vehicle_log.json"
    
    if not config_path.exists() or not log_path.exists():
        print(f"Skipping {run_dir.name}: missing config.json or vehicle_log.json")
        return 0
    
    config = load_config(config_path)
    frame_lookup = load_vehicle_log(log_path)
    
    run_output_dir = output_dir / run_dir.name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    for frame_id, frame_data in tqdm(frame_lookup.items(), desc=f"Processing {run_dir.name}"):
        try:
            sample = process_frame(run_dir, frame_id, frame_data, config, transform)
            if sample is not None:
                output_path = run_output_dir / f"{frame_id:06d}.pt"
                torch.save(sample, output_path)
                processed_count += 1
        except Exception as e:
            print(f"Error processing frame {frame_id} in {run_dir.name}: {e}")
            continue
    
    return processed_count

def main():
    parser = argparse.ArgumentParser(description="Preprocess CARLA data for MoE training")
    parser.add_argument("--raw_dir", type=str, default=RAW_DIR, 
                        help="Directory containing raw CARLA runs")
    parser.add_argument("--out_dir", type=str, default=PREPROCESSED_DIR,
                        help="Output directory for preprocessed data")
    parser.add_argument("--runs", type=int, nargs="*", 
                        help="Specific runs to process (e.g., 1 2 3). If not specified, processes all runs")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Train/val split ratio (default: 0.8 for training)")
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    
    if not raw_dir.exists():
        print(f"Error: Raw directory {raw_dir} does not exist")
        return
    
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    if args.runs:
        runs_to_process = [f"run_{run_num:03d}" for run_num in args.runs]
    else:
        runs_to_process = [d.name for d in raw_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('run_')]
    
    runs_to_process.sort()
    
    print(f"Found {len(runs_to_process)} runs to process")
    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Train/val split: {args.split_ratio:.1%} / {1-args.split_ratio:.1%}")
    
    transform = get_transforms()
    
    split_idx = int(len(runs_to_process) * args.split_ratio)
    train_runs = runs_to_process[:split_idx]
    val_runs = runs_to_process[split_idx:]
    
    print(f"\nTrain runs ({len(train_runs)}): {train_runs}")
    print(f"Val runs ({len(val_runs)}): {val_runs}")
    
    total_train_frames = 0
    print(f"\n🚂 Processing training runs...")
    for run_name in train_runs:
        run_path = raw_dir / run_name
        if run_path.exists():
            count = process_run(run_path, train_dir, transform)
            total_train_frames += count
            print(f"✅ {run_name}: {count} frames processed")
        else:
            print(f"⚠️  {run_name}: directory not found")
    
    total_val_frames = 0
    print(f"\n🔍 Processing validation runs...")
    for run_name in val_runs:
        run_path = raw_dir / run_name
        if run_path.exists():
            count = process_run(run_path, val_dir, transform)
            total_val_frames += count
            print(f"✅ {run_name}: {count} frames processed")
        else:
            print(f"⚠️  {run_name}: directory not found")
    
    print(f"\n📊 Preprocessing complete!")
    print(f"   Training frames: {total_train_frames}")
    print(f"   Validation frames: {total_val_frames}")
    print(f"   Total frames: {total_train_frames + total_val_frames}")
    print(f"   Output directory: {out_dir}")

if __name__ == "__main__":
    main()