from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
import os

VERSION = os.getenv('NUSC_VERSION', 'v1.0-trainval')
DATAROOT = os.getenv('NUSC_DATAROOT', 'datasets/nuscenes/raw')

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

def filter_available_scenes(nusc):
    available_scenes = []
    print(f"Checking {len(nusc.scene)} scenes for available files...")
    
    for scene in tqdm(nusc.scene, desc="Processing scenes", unit="scene"):
        sample_token = scene['first_sample_token']
        sample = nusc.get('sample', sample_token)
        cam_token = sample['data']['CAM_FRONT']
        cam_path, _, _ = nusc.get_sample_data(cam_token)
        try:
            with open(cam_path, 'rb'):
                available_scenes.append(scene)
        except FileNotFoundError:
            continue
    return available_scenes

available_scenes = filter_available_scenes(nusc)
print(f"\nFound {len(available_scenes)} available scenes out of {len(nusc.scene)} total scenes")