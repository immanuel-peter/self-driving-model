"""
Gating Network for Mixture of Experts (MoE) model.
This implements Step 3 of the AutoMoE roadmap: Train gating mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional


class GatingNetwork(nn.Module):
    """
    Gating network that learns to select or weight expert outputs based on context.
    
    Supports both hard gating (select one expert) and soft gating (weighted combination).
    """
    
    def __init__(self, 
                 num_experts: int = 4,
                 context_dim: int = 512,
                 gating_type: str = 'soft',
                 image_backbone: str = 'resnet18',
                 use_weather_context: bool = True,
                 use_traffic_context: bool = True):
        """
        Args:
            num_experts: Number of expert models (4: BDD Detection, BDD Seg, BDD Drivable, NuScenes)
            context_dim: Dimension of context features
            gating_type: 'soft' for weighted combination, 'hard' for expert selection
            image_backbone: Backbone for image feature extraction
            use_weather_context: Whether to use weather information
            use_traffic_context: Whether to use traffic density information
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.context_dim = context_dim
        self.gating_type = gating_type
        self.use_weather_context = use_weather_context
        self.use_traffic_context = use_traffic_context
        
        # Image feature extractor
        self.setup_image_backbone(image_backbone)
        
        # Context encoders
        self.setup_context_encoders()
        
        # Gating head
        self.setup_gating_head()
        
        # Expert names for logging/debugging
        self.expert_names = ['bdd_detection', 'bdd_segmentation', 'bdd_drivable', 'nuscenes']
        
    def setup_image_backbone(self, backbone_name: str):
        """Setup image feature extraction backbone"""
        if backbone_name == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            self.image_backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.image_proj = nn.Linear(512, self.context_dim)
        elif backbone_name == 'resnet34':
            resnet = models.resnet34(pretrained=True)
            self.image_backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.image_proj = nn.Linear(512, self.context_dim)
        elif backbone_name == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.image_backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.image_proj = nn.Linear(2048, self.context_dim)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    def setup_context_encoders(self):
        """Setup additional context encoders"""
        self.context_encoders = nn.ModuleDict()
        
        if self.use_weather_context:
            # Weather encoder (categorical + continuous)
            # Assume weather categories: Clear, Rain, Snow, Fog, etc.
            self.weather_categories = 8  # Adjust based on your weather types
            self.weather_embedding = nn.Embedding(self.weather_categories, 64)
            self.weather_proj = nn.Linear(64 + 3, 128)  # +3 for visibility, wetness, etc.
            self.context_encoders['weather'] = nn.Sequential(
                self.weather_proj,
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        
        if self.use_traffic_context:
            # Traffic context encoder
            self.context_encoders['traffic'] = nn.Sequential(
                nn.Linear(4, 64),  # vehicle_count, pedestrian_count, traffic_light_state, speed_limit
                nn.ReLU(),
                nn.Linear(64, 64)
            )
    
    def setup_gating_head(self):
        """Setup the final gating decision head"""
        # Calculate total context dimension
        total_context_dim = self.context_dim  # Image features
        
        if self.use_weather_context:
            total_context_dim += 64
        if self.use_traffic_context:
            total_context_dim += 64
        
        # Gating MLP
        self.gating_head = nn.Sequential(
            nn.Linear(total_context_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_experts)
        )
    
    def extract_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features from input image"""
        # image: [B, 3, H, W]
        with torch.set_grad_enabled(self.training):
            features = self.image_backbone(image)  # [B, feat_dim, 1, 1]
            features = features.view(features.size(0), -1)  # [B, feat_dim]
            features = self.image_proj(features)  # [B, context_dim]
        return features
    
    def encode_weather_context(self, weather_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode weather context"""
        if not self.use_weather_context:
            return None
        
        # Weather category embedding
        weather_cat = weather_data.get('category', torch.zeros(weather_data['visibility'].size(0), dtype=torch.long, device=weather_data['visibility'].device))
        weather_emb = self.weather_embedding(weather_cat)  # [B, 64]
        
        # Weather continuous features
        weather_cont = torch.stack([
            weather_data.get('visibility', torch.ones_like(weather_cat, dtype=torch.float)),
            weather_data.get('wetness', torch.zeros_like(weather_cat, dtype=torch.float)),
            weather_data.get('cloudiness', torch.zeros_like(weather_cat, dtype=torch.float))
        ], dim=1)  # [B, 3]
        
        # Combine embedding and continuous features
        weather_combined = torch.cat([weather_emb, weather_cont], dim=1)  # [B, 67]
        weather_encoded = self.context_encoders['weather'](weather_combined)  # [B, 64]
        
        return weather_encoded
    
    def encode_traffic_context(self, traffic_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode traffic context"""
        if not self.use_traffic_context:
            return None
        
        # Traffic features
        traffic_features = torch.stack([
            traffic_data.get('vehicle_count', torch.zeros_like(traffic_data.get('speed_limit', torch.zeros(1)))),
            traffic_data.get('pedestrian_count', torch.zeros_like(traffic_data.get('speed_limit', torch.zeros(1)))),
            traffic_data.get('traffic_light_state', torch.zeros_like(traffic_data.get('speed_limit', torch.zeros(1)))),  # 0=Red, 1=Yellow, 2=Green
            traffic_data.get('speed_limit', torch.ones_like(traffic_data.get('speed_limit', torch.ones(1))) * 50.0)  # Default 50 km/h
        ], dim=1)  # [B, 4]
        
        traffic_encoded = self.context_encoders['traffic'](traffic_features)  # [B, 64]
        return traffic_encoded
    
    def forward(self, 
                image: torch.Tensor,
                weather_data: Optional[Dict[str, torch.Tensor]] = None,
                traffic_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of gating network
        
        Args:
            image: Input image [B, 3, H, W]
            weather_data: Dictionary with weather information
            traffic_data: Dictionary with traffic information
            
        Returns:
            Dictionary containing:
            - 'expert_weights': Weights for each expert [B, num_experts]
            - 'expert_probs': Softmax probabilities [B, num_experts]
            - 'selected_expert': Hard selection indices [B] (if hard gating)
        """
        batch_size = image.size(0)
        
        # Extract image features
        image_features = self.extract_image_features(image)  # [B, context_dim]
        
        # Collect all context features
        context_features = [image_features]
        
        # Weather context
        if self.use_weather_context and weather_data is not None:
            weather_features = self.encode_weather_context(weather_data)
            context_features.append(weather_features)
        
        # Traffic context
        if self.use_traffic_context and traffic_data is not None:
            traffic_features = self.encode_traffic_context(traffic_data)
            context_features.append(traffic_features)
        
        # Concatenate all context features
        combined_context = torch.cat(context_features, dim=1)  # [B, total_context_dim]
        
        # Gating decision
        expert_logits = self.gating_head(combined_context)  # [B, num_experts]
        expert_probs = F.softmax(expert_logits, dim=1)  # [B, num_experts]
        
        # Prepare output
        output = {
            'expert_weights': expert_logits,
            'expert_probs': expert_probs,
            'context_features': combined_context
        }
        
        if self.gating_type == 'hard':
            # Hard gating: select one expert
            selected_expert = torch.argmax(expert_probs, dim=1)  # [B]
            output['selected_expert'] = selected_expert
        
        return output
    
    def get_expert_selection_stats(self, 
                                  image: torch.Tensor,
                                  weather_data: Optional[Dict[str, torch.Tensor]] = None,
                                  traffic_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """Get statistics about expert selection"""
        with torch.no_grad():
            gating_output = self.forward(image, weather_data, traffic_data)
            expert_probs = gating_output['expert_probs']  # [B, num_experts]
            
            # Calculate selection statistics
            stats = {}
            for i, expert_name in enumerate(self.expert_names):
                avg_weight = expert_probs[:, i].mean().item()
                max_weight = expert_probs[:, i].max().item()
                selection_count = (expert_probs.argmax(dim=1) == i).sum().item()
                
                stats[f'{expert_name}_avg_weight'] = avg_weight
                stats[f'{expert_name}_max_weight'] = max_weight
                stats[f'{expert_name}_selection_count'] = selection_count
                stats[f'{expert_name}_selection_rate'] = selection_count / image.size(0)
            
            return stats


class MoEModel(nn.Module):
    """
    Complete Mixture of Experts model combining experts with gating network
    """
    
    def __init__(self, 
                 experts: Dict[str, nn.Module],
                 gating_network: GatingNetwork,
                 output_dim: int = 2,  # e.g., steering + throttle
                 freeze_experts: bool = True):
        """
        Args:
            experts: Dictionary of expert models
            gating_network: Gating network
            output_dim: Dimension of final output (e.g., 2 for steering + throttle)
            freeze_experts: Whether to freeze expert weights during training
        """
        super().__init__()
        
        self.experts = nn.ModuleDict(experts)
        self.gating_network = gating_network
        self.output_dim = output_dim
        self.freeze_experts = freeze_experts
        
        if freeze_experts:
            # Freeze expert parameters
            for expert in self.experts.values():
                for param in expert.parameters():
                    param.requires_grad = False
        
        # Output projection layers to standardize expert outputs
        self.expert_projections = nn.ModuleDict()
        for expert_name in experts.keys():
            # Assume each expert outputs some feature dimension - project to output_dim
            self.expert_projections[expert_name] = nn.Linear(self.get_expert_output_dim(expert_name), output_dim)
    
    def get_expert_output_dim(self, expert_name: str) -> int:
        """Get output dimension of each expert (you may need to adjust this)"""
        # This is a placeholder - you'll need to check actual expert output dims
        if expert_name == 'nuscenes':
            return 10  # As defined in NuScenesExpert
        else:
            return 128  # Placeholder for BDD experts
    
    def forward(self, 
                image: torch.Tensor,
                lidar: Optional[List[torch.Tensor]] = None,
                weather_data: Optional[Dict[str, torch.Tensor]] = None,
                traffic_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of complete MoE model
        
        Args:
            image: Input image [B, 3, H, W]
            lidar: LiDAR data for NuScenes expert
            weather_data: Weather context
            traffic_data: Traffic context
            
        Returns:
            Dictionary with final output and gating information
        """
        batch_size = image.size(0)
        
        # Get gating decisions
        gating_output = self.gating_network(image, weather_data, traffic_data)
        expert_weights = gating_output['expert_probs']  # [B, num_experts]
        
        # Get outputs from all experts
        expert_outputs = {}
        
        # BDD Detection Expert
        if 'bdd_detection' in self.experts:
            detection_out = self.experts['bdd_detection'](image)
            if isinstance(detection_out, dict):
                # Flatten detection output for projection
                detection_features = torch.cat([
                    detection_out['class_logits'].mean(dim=[2, 3]),  # Global average pooling
                    detection_out['bbox_deltas'].mean(dim=[2, 3])
                ], dim=1)
            else:
                detection_features = detection_out
            expert_outputs['bdd_detection'] = self.expert_projections['bdd_detection'](detection_features)
        
        # BDD Segmentation Expert
        if 'bdd_segmentation' in self.experts:
            seg_out = self.experts['bdd_segmentation'](image)
            if len(seg_out.shape) == 4:  # [B, C, H, W]
                seg_features = seg_out.mean(dim=[2, 3])  # Global average pooling
            else:
                seg_features = seg_out
            expert_outputs['bdd_segmentation'] = self.expert_projections['bdd_segmentation'](seg_features)
        
        # BDD Drivable Expert
        if 'bdd_drivable' in self.experts:
            drivable_out = self.experts['bdd_drivable'](image)
            if len(drivable_out.shape) == 4:  # [B, C, H, W]
                drivable_features = drivable_out.mean(dim=[2, 3])  # Global average pooling
            else:
                drivable_features = drivable_out
            expert_outputs['bdd_drivable'] = self.expert_projections['bdd_drivable'](drivable_features)
        
        # NuScenes Expert
        if 'nuscenes' in self.experts:
            # NuScenes expert needs both image and lidar
            nuscenes_input = {'image': image}
            if lidar is not None:
                nuscenes_input['lidar'] = lidar
            else:
                # Create dummy lidar if not provided
                nuscenes_input['lidar'] = [torch.zeros(100, 3, device=image.device) for _ in range(batch_size)]
            
            nuscenes_out = self.experts['nuscenes'](nuscenes_input)
            expert_outputs['nuscenes'] = self.expert_projections['nuscenes'](nuscenes_out)
        
        # Combine expert outputs based on gating weights
        if self.gating_network.gating_type == 'soft':
            # Soft gating: weighted combination
            combined_output = torch.zeros(batch_size, self.output_dim, device=image.device)
            
            for i, expert_name in enumerate(self.gating_network.expert_names):
                if expert_name in expert_outputs:
                    weight = expert_weights[:, i:i+1]  # [B, 1]
                    combined_output += weight * expert_outputs[expert_name]
        
        else:
            # Hard gating: select one expert
            selected_experts = gating_output['selected_expert']  # [B]
            combined_output = torch.zeros(batch_size, self.output_dim, device=image.device)
            
            for i, expert_name in enumerate(self.gating_network.expert_names):
                if expert_name in expert_outputs:
                    mask = (selected_experts == i).float().unsqueeze(1)  # [B, 1]
                    combined_output += mask * expert_outputs[expert_name]
        
        return {
            'final_output': combined_output,
            'expert_weights': expert_weights,
            'expert_outputs': expert_outputs,
            'gating_output': gating_output
        }
