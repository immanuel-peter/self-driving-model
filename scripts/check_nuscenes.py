from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
import os

VERSION = os.getenv('NUSC_VERSION', 'v1.0-trainval')
DATAROOT = os.getenv('NUSC_DATAROOT', 'datasets/nuscenes/raw')

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

# Assume only three subsets have been downloaded

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

"""
Output:

======
Loading NuScenes tables for version v1.0-trainval...
23 category,
8 attribute,
4 visibility,
64386 instance,
12 sensor,
10200 calibrated_sensor,
2631083 ego_pose,
68 log,
850 scene,
34149 sample,
2631083 sample_data,
1166187 sample_annotation,
4 map,
Done loading in 9.760 seconds.
======
Reverse indexing ...
Done reverse indexing in 3.1 seconds.
======
Checking 850 scenes for available files...
Processing scenes: 100%|████████████████████████████████████████████████████████████████████| 850/850 [00:01<00:00, 534.17scene/s]

Found 255 available scenes out of 850 total scenes
"""