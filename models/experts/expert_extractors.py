import torch
import torch.nn as nn
from typing import Dict, List

class ExpertOutputExtractor(nn.Module):
    """Base class for extracting features from expert outputs"""
    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        
    def forward(self, expert_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            expert_output: Dict containing expert-specific outputs
        Returns:
            features: [B, output_dim] - extracted features
        """
        raise NotImplementedError

class DetectionExpertExtractor(ExpertOutputExtractor):
    """Extracts features from detection expert outputs"""
    def __init__(self, output_dim: int = 256, num_classes: int = 10):
        super().__init__(output_dim)
        self.num_classes = num_classes
        
        # Global average pooling + MLP to extract features
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(num_classes + 4, 512),  # class_logits + bbox_deltas
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, expert_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            expert_output: Dict with 'class_logits' [B, C, H, W] and 'bbox_deltas' [B, 4, H, W]
        Returns:
            features: [B, output_dim] - extracted features
        """
        class_logits = expert_output['class_logits']  # [B, C, H, W]
        bbox_deltas = expert_output['bbox_deltas']    # [B, 4, H, W]
        
        # Concatenate along channel dimension
        combined = torch.cat([class_logits, bbox_deltas], dim=1)  # [B, C+4, H, W]
        
        # Extract features
        features = self.feature_extractor(combined)  # [B, output_dim]
        return features

class SegmentationExpertExtractor(ExpertOutputExtractor):
    """Extracts features from segmentation expert outputs"""
    def __init__(self, output_dim: int = 256, num_classes: int = 19):
        super().__init__(output_dim)
        self.num_classes = num_classes
        
        # Global average pooling + MLP to extract features
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(num_classes, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, expert_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_output: [B, num_classes, H, W] - segmentation logits
        Returns:
            features: [B, output_dim] - extracted features
        """
        # Extract features
        features = self.feature_extractor(expert_output)  # [B, output_dim]
        return features

class DrivableExpertExtractor(ExpertOutputExtractor):
    """Extracts features from drivable area expert outputs"""
    def __init__(self, output_dim: int = 256, num_classes: int = 3):
        super().__init__(output_dim)
        self.num_classes = num_classes
        
        # Global average pooling + MLP to extract features
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(num_classes, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, expert_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_output: [B, num_classes, H, W] - drivable area logits
        Returns:
            features: [B, output_dim] - extracted features
        """
        # Extract features
        features = self.feature_extractor(expert_output)  # [B, output_dim]
        return features

class NuScenesExpertExtractor(ExpertOutputExtractor):
    """Extracts features from nuScenes expert outputs"""
    def __init__(self, output_dim: int = 256, num_queries: int = 100, num_classes: int = 10, bbox_dim: int = 7):
        super().__init__(output_dim)
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.bbox_dim = bbox_dim
        
        # Process query-based outputs
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_queries * (num_classes + self.bbox_dim), 512),  # class_logits + bbox_preds
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, expert_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            expert_output: Dict with 'class_logits' [B, Q, C] and 'bbox_preds' [B, Q, bbox_dim]
        Returns:
            features: [B, output_dim] - extracted features
        """
        class_logits = expert_output['class_logits']  # [B, Q, C]
        bbox_preds = expert_output['bbox_preds']      # [B, Q, 7]
        
        # Concatenate along last dimension and flatten
        combined = torch.cat([class_logits, bbox_preds], dim=-1)  # [B, Q, C+bbox_dim]
        flattened = combined.view(combined.size(0), -1)  # [B, Q*(C+bbox_dim)]
        
        # Extract features
        features = self.feature_extractor(flattened)  # [B, output_dim]
        return features


class ExpertOutputManager(nn.Module):
    """Manages multiple expert output extractors as a registered module"""
    def __init__(self, extractors: List[ExpertOutputExtractor]):
        super().__init__()
        self.extractors = nn.ModuleList(extractors)
        
    def extract_features(self, expert_outputs: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        """
        Args:
            expert_outputs: List of expert outputs (each can be Dict or Tensor)
        Returns:
            features: List of [B, output_dim] tensors
        """
        features = []
        for extractor, expert_output in zip(self.extractors, expert_outputs):
            feature = extractor(expert_output)
            features.append(feature)
        return features

def create_expert_extractors(expert_configs: List[Dict]) -> ExpertOutputManager:
    """
    Create expert extractors based on configuration
    
    Args:
        expert_configs: List of dicts with 'type' and other parameters
    Returns:
        ExpertOutputManager with configured extractors
    """
    extractors = []
    
    for config in expert_configs:
        expert_type = config['type']
        
        if expert_type == 'detection':
            extractor = DetectionExpertExtractor(
                output_dim=config.get('output_dim', 256),
                num_classes=config.get('num_classes', 10)
            )
        elif expert_type == 'segmentation':
            extractor = SegmentationExpertExtractor(
                output_dim=config.get('output_dim', 256),
                num_classes=config.get('num_classes', 19)
            )
        elif expert_type == 'drivable':
            extractor = DrivableExpertExtractor(
                output_dim=config.get('output_dim', 256),
                num_classes=config.get('num_classes', 3)
            )
        elif expert_type == 'nuscenes':
            extractor = NuScenesExpertExtractor(
                output_dim=config.get('output_dim', 256),
                num_queries=config.get('num_queries', 100),
                num_classes=config.get('num_classes', 10),
                bbox_dim=config.get('bbox_dim', 7)
            )
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")
            
        extractors.append(extractor)
    
    return ExpertOutputManager(extractors)

