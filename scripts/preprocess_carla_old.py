import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torchvision import transforms

RAW_DIR = "datasets/carla/raw"
PREPROCESSED_DIR = "datasets/carla/preprocessed"
CAMERA_CONFIGS = ["front"]

# Image preprocessing transforms
def get_transforms():
    """Get image preprocessing transforms similar to other datasets"""
    return transforms.Compose([
        transforms.Resize((256, 256)),  # Consistent with NuScenes preprocessing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_vehicle_log(log_path):
    """Load and index vehicle log by frame number"""
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    # Create a lookup dict by frame number
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
    
    # Load images from all cameras
    images = {}
    image_filename = frame_data['image_filename']
    
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
    
    # Skip frame if we don't have the main front camera
    if 'front' not in images:
        return None
    
    # Extract vehicle state information
    location = frame_data['location']
    rotation = frame_data['rotation'] 
    velocity = frame_data['velocity']
    control = frame_data['control']
    traffic = frame_data['traffic_density']
    
    # Weather information
    weather_features = parse_weather_info(config['weather'])
    
    # Create the sample structure compatible with MoE experts
    sample = {
        # Front camera image
        'image': images['front'],
        
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
                weather_features['cloudiness'] / 100.0,  # Normalize
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
            'camera': 'front'
        }
    }
    
    return sample

def process_run(run_dir, output_dir, transform):
    """Process a single CARLA run"""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    
    # Load configuration and vehicle log
    config_path = run_dir / "config.json"
    log_path = run_dir / "vehicle_log.json"
    
    if not config_path.exists() or not log_path.exists():
        print(f"Skipping {run_dir.name}: missing config.json or vehicle_log.json")
        return 0
    
    config = load_config(config_path)
    frame_lookup = load_vehicle_log(log_path)
    
    # Create output directory for this run
    run_output_dir = output_dir / run_dir.name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    # Process each frame in the vehicle log
    for frame_id, frame_data in tqdm(frame_lookup.items(), desc=f"Processing {run_dir.name}"):
        try:
            sample = process_frame(run_dir, frame_id, frame_data, config, transform)
            if sample is not None:
                # Save as .pt file
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
    
    # Create output directories
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of runs to process
    if args.runs:
        runs_to_process = [f"run_{run_num:03d}" for run_num in args.runs]
    else:
        runs_to_process = [d.name for d in raw_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('run_')]
    
    runs_to_process.sort()  # Ensure consistent ordering
    
    print(f"Found {len(runs_to_process)} runs to process")
    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Train/val split: {args.split_ratio:.1%} / {1-args.split_ratio:.1%}")
    
    # Initialize transforms
    transform = get_transforms()
    
    # Split runs into train/val
    split_idx = int(len(runs_to_process) * args.split_ratio)
    train_runs = runs_to_process[:split_idx]
    val_runs = runs_to_process[split_idx:]
    
    print(f"\nTrain runs ({len(train_runs)}): {train_runs}")
    print(f"Val runs ({len(val_runs)}): {val_runs}")
    
    # Process training runs
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
    
    # Process validation runs
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
    
    # Summary
    print(f"\n📊 Preprocessing complete!")
    print(f"   Training frames: {total_train_frames}")
    print(f"   Validation frames: {total_val_frames}")
    print(f"   Total frames: {total_train_frames + total_val_frames}")
    print(f"   Output directory: {out_dir}")

if __name__ == "__main__":
    main()