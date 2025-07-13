# Import train/val loaders
from .bdd_detection_loader import get_train_loader as get_bdd_detection_train_loader
from .bdd_detection_loader import get_val_loader as get_bdd_detection_val_loader

from .bdd_segmentation_loader import get_train_loader as get_bdd_segmentation_train_loader
from .bdd_segmentation_loader import get_val_loader as get_bdd_segmentation_val_loader

from .bdd_drivable_loader import get_train_loader as get_bdd_drivable_train_loader
from .bdd_drivable_loader import get_val_loader as get_bdd_drivable_val_loader

from .nuscenes_loader import get_train_loader as get_nuscenes_train_loader
from .nuscenes_loader import get_val_loader as get_nuscenes_val_loader

# Export ready-to-use loaders for training scripts
bdd_detection_train_loader = get_bdd_detection_train_loader()
bdd_detection_val_loader = get_bdd_detection_val_loader()

bdd_segmentation_train_loader = get_bdd_segmentation_train_loader()
bdd_segmentation_val_loader = get_bdd_segmentation_val_loader()

bdd_drivable_train_loader = get_bdd_drivable_train_loader()
bdd_drivable_val_loader = get_bdd_drivable_val_loader()

nuscenes_train_loader = get_nuscenes_train_loader()
nuscenes_val_loader = get_nuscenes_val_loader()

__all__ = [
    # BDD100K Detection
    "bdd_detection_train_loader",
    "bdd_detection_val_loader",
    
    # BDD100K Segmentation
    "bdd_segmentation_train_loader", 
    "bdd_segmentation_val_loader",
    
    # BDD100K Drivable Area
    "bdd_drivable_train_loader",
    "bdd_drivable_val_loader",
    
    # NuScenes Multimodal
    "nuscenes_train_loader",
    "nuscenes_val_loader",
]