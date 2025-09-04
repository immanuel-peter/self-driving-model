import os
import torch
from tqdm import tqdm
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from torchvision import transforms

VERSION = os.getenv('NUSC_VERSION', 'v1.0-trainval')
DATAROOT = os.getenv('NUSC_DATAROOT', 'datasets/nuscenes/raw')
CACHE_DIR = 'datasets/nuscenes/preprocessed/'


def get_scenes_by_split(nusc):
    """
    Split NuScenes scenes into train/val using official NuScenes splits.
    """
    splits = create_splits_scenes()
    train_scene_names = set(splits['train'])
    val_scene_names = set(splits['val'])
    
    available = {'train': [], 'val': []}
    
    for scene in nusc.scene:
        name = scene['name']
        if name in train_scene_names:
            available['train'].append(scene)
        elif name in val_scene_names:
            available['val'].append(scene)
        else:
            available['train'].append(scene)

    return available


def cache_sample(nusc, sample_token, transform, out_dir):
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data']['CAM_FRONT']
    lidar_token = sample['data']['LIDAR_TOP']

    try:
        cam_path, _, cam_intrinsics = nusc.get_sample_data(cam_token)
        image = Image.open(cam_path).convert('RGB')
        image = transform(image)

        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        lidar = LidarPointCloud.from_file(lidar_path).points.T[:, :3]  # (N, 3)

        out = {
            'image': image,
            'lidar': torch.tensor(lidar, dtype=torch.float32),
            'boxes': boxes,
            'intrinsics': torch.tensor(cam_intrinsics, dtype=torch.float32),
            'token': sample_token
        }

        torch.save(out, os.path.join(out_dir, f'{sample_token}.pt'))

    except FileNotFoundError:
        raise FileNotFoundError("File not found")

def build_cache():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    split_scenes = get_scenes_by_split(nusc)

    for split in ['train', 'val']:
        split_dir = os.path.join(CACHE_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

        print(f"Processing split: {split}")
        for scene in tqdm(split_scenes[split]):
            token = scene['first_sample_token']
            while token:
                sample = nusc.get('sample', token)
                try:
                    cache_sample(nusc, token, transform, out_dir=split_dir)
                except FileNotFoundError:
                    pass
                token = sample['next']


if __name__ == '__main__':
    build_cache()