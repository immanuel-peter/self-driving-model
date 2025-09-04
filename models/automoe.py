import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import warnings

from .experts import BDDDetectionExpert, BDDDrivableExpert, BDDSegmentationExpert, NuScenesExpert
from .policy.trajectory_head import TrajectoryPolicy
from .gating.gating_network import GatingNetwork
from .experts.expert_extractors import create_expert_extractors
from .context.context_features import create_context_extractor

class AutoMoE(nn.Module):
    """Complete AutoMoE: Mixture of Experts Self-Driving Model"""
    
    def __init__(self, 
                 expert_configs: List[Dict],
                 gating_config: Dict,
                 context_config: Dict,
                 policy_config: Dict,
                 device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.expert_configs = expert_configs
        self.gating_config = gating_config
        self.context_config = context_config
        self.policy_config = policy_config
        
        self.experts = self._create_experts()
        
        self.expert_extractors = create_expert_extractors(expert_configs)
        
        self.context_extractor = create_context_extractor(context_config)
        
        self.gating_network = self._create_gating_network()
        
        self.policy_head = self._create_policy_head()
        
        self.to(device)
        
    def _create_experts(self) -> nn.ModuleList:
        """Create expert models based on configuration"""
        experts = nn.ModuleList()
        
        for config in self.expert_configs:
            expert_type = config['type']
            
            if expert_type == 'detection':
                expert = BDDDetectionExpert(
                    num_classes=config.get('num_classes', 10),
                    pretrained_backbone=config.get('pretrained_backbone', True)
                )
            elif expert_type == 'segmentation':
                expert = BDDSegmentationExpert(
                    num_classes=config.get('num_classes', 19),
                    pretrained_backbone=config.get('pretrained_backbone', True)
                )
            elif expert_type == 'drivable':
                expert = BDDDrivableExpert(
                    num_classes=config.get('num_classes', 3),
                    pretrained_backbone=config.get('pretrained_backbone', True)
                )
            elif expert_type == 'nuscenes':
                expert = NuScenesExpert(
                    num_queries=config.get('num_queries', 100),
                    fusion=config.get('fusion', 'concat'),
                    use_lidar=config.get('use_lidar', False),
                    use_tnet=config.get('use_tnet', False),
                    bbox_dim=config.get('bbox_dim', 7)
                )
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")
            
            experts.append(expert)
            
        return experts
    
    def _create_gating_network(self) -> GatingNetwork:
        """Create gating network based on configuration"""
        num_experts = len(self.expert_configs)
        expert_output_dims = [config.get('output_dim', 256) for config in self.expert_configs]
        
        return GatingNetwork(
            num_experts=num_experts,
            context_dim=self.context_config.get('context_dim', 64),
            expert_output_dims=expert_output_dims,
            processed_dim=self.gating_config.get('processed_dim', 256),
            hidden_dim=self.gating_config.get('hidden_dim', 128),
            temperature=self.gating_config.get('temperature', 1.0),
            use_softmax=self.gating_config.get('use_softmax', True)
        )
    
    def _create_policy_head(self) -> TrajectoryPolicy:
        """Create policy head based on configuration"""
        return TrajectoryPolicy(
            horizon=self.policy_config.get('num_waypoints', 10),
            context_dim=self.gating_config.get('processed_dim', 256),
            backbone_dim=self.policy_config.get('backbone_dim', 512)
        )
    
    def _extract_context_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract context features from batch"""
        # Fallbacks: dataset may not include steering/throttle/brake; build a minimal context
        has_simple_keys = all(k in batch for k in ('speed', 'steering', 'throttle', 'brake'))

        if self.context_config.get('type', 'simple') == 'simple':
            # Simple context extractor expects [B,1] tensors
            if batch['speed'].dim() == 2 and batch['speed'].size(1) > 1:
                speed_in = batch['speed'][:, -1:].contiguous()
            else:
                speed_in = batch['speed']
            if not has_simple_keys:
                bsz = speed_in.size(0)
                device = speed_in.device
                steering = torch.zeros(bsz, 1, device=device)
                throttle = torch.zeros(bsz, 1, device=device)
                brake = torch.zeros(bsz, 1, device=device)
            else:
                # Ensure shapes use last-step [B,1] consistently
                steering = batch['steering']
                throttle = batch['throttle']
                brake = batch['brake']
                if steering.dim() == 2 and steering.size(1) > 1:
                    steering = steering[:, -1:].contiguous()
                elif steering.dim() > 2:
                    steering = steering.view(steering.size(0), -1)[:, -1:].contiguous()
                if throttle.dim() == 2 and throttle.size(1) > 1:
                    throttle = throttle[:, -1:].contiguous()
                elif throttle.dim() > 2:
                    throttle = throttle.view(throttle.size(0), -1)[:, -1:].contiguous()
                if brake.dim() == 2 and brake.size(1) > 1:
                    brake = brake[:, -1:].contiguous()
                elif brake.dim() > 2:
                    brake = brake.view(brake.size(0), -1)[:, -1:].contiguous()
            return self.context_extractor(speed_in, steering, throttle, brake)
        else:
            # Full context extractor; create zeros for missing fields
            bsz = batch['speed'].size(0)
            device = batch['speed'].device
            if batch['speed'].dim() == 2 and batch['speed'].size(1) > 1:
                speed_in = batch['speed'][:, -1:].contiguous()
            else:
                speed_in = batch['speed']
            context_data = {
                'speed': speed_in,
                'steering': batch.get('steering', torch.zeros(bsz, 1, device=device)),
                'throttle': batch.get('throttle', torch.zeros(bsz, 1, device=device)),
                'brake': batch.get('brake', torch.zeros(bsz, 1, device=device)),
                'hour': batch.get('hour', torch.zeros(bsz, 1, device=device)),
                'minute': batch.get('minute', torch.zeros(bsz, 1, device=device)),
                'weather': batch.get('weather', {}),
                'road': batch.get('road', {})
            }
            return self.context_extractor(context_data)
    
    def _run_experts(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Run all experts and extract features"""
        expert_outputs = []
        
        for i, expert in enumerate(self.experts):
            expert_type = self.expert_configs[i]['type']
            
            try:
                if expert_type == 'detection':
                    expert_output = expert(batch['image'])
                elif expert_type == 'segmentation':
                    expert_output = expert(batch['image'])
                elif expert_type == 'drivable':
                    expert_output = expert(batch['image'])
                elif expert_type == 'nuscenes':
                    expert_input = {
                        'image': batch['image'],
                        'lidar': batch.get('lidar', torch.zeros(batch['image'].size(0), 1000, 3, device=self.device))
                    }
                    expert_output = expert(expert_input)
                else:
                    raise ValueError(f"Unknown expert type: {expert_type}")
                
                expert_outputs.append(expert_output)
                
            except Exception as e:
                warnings.warn(f"Error running expert {i} ({expert_type}): {str(e)}")
                batch_size = batch['image'].size(0)
                dummy_output = torch.zeros(batch_size, self.expert_configs[i].get('output_dim', 256), device=self.device)
                expert_outputs.append(dummy_output)
        
        return expert_outputs
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete AutoMoE model
        
        Args:
            batch: Dict containing:
                - image: [B, C, H, W] - camera image
                - lidar: [B, N, 3] - lidar point cloud (optional)
                - speed: [B, 1] - vehicle speed
                - steering: [B, 1] - steering angle
                - throttle: [B, 1] - throttle input
                - brake: [B, 1] - brake input
                - Additional context features (weather, time, road, etc.)
        Returns:
            Dict containing:
                - waypoints: [B, num_waypoints, waypoint_dim] - predicted waypoints
                - speed: [B, 1] - predicted speed
                - expert_weights: [B, num_experts] - gating weights
                - expert_outputs: List of expert outputs
                - context_features: [B, context_dim] - encoded context
        """
        context_features = self._extract_context_features(batch)
        
        expert_outputs = self._run_experts(batch)
        
        expert_features = self.expert_extractors.extract_features(expert_outputs)
        
        gating_output = self.gating_network(expert_features, context_features)
        
        policy_output = self.policy_head(batch['image'], context=gating_output['combined_output'])
        speed_seq = policy_output.get('speed')
        speed_out = None
        if speed_seq is not None and speed_seq.dim() == 2:
            speed_out = speed_seq[:, -1:].contiguous()
        
        return {
            'waypoints': policy_output['waypoints'],
            'speed': speed_out if speed_out is not None else speed_seq,
            'speed_seq': speed_seq,
            'expert_weights': gating_output['expert_weights'],
            'expert_outputs': expert_outputs,
            'context_features': context_features,
            'combined_features': gating_output['combined_output'],
            'gate_logits': gating_output['gate_logits']
        }
    
    def get_expert_weights(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get expert weights without running experts (for analysis)"""
        context_features = self._extract_context_features(batch)
        return self.gating_network.get_expert_weights(context_features)
    
    def load_expert_checkpoints(self, checkpoint_paths: List[str]):
        """Load pre-trained expert checkpoints"""
        if len(checkpoint_paths) != len(self.experts):
            raise ValueError(f"Expected {len(self.experts)} checkpoint paths, got {len(checkpoint_paths)}")
        
        for i, (expert, checkpoint_path) in enumerate(zip(self.experts, checkpoint_paths)):
            if checkpoint_path and checkpoint_path != '':
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    state_dict = checkpoint.get('model_state_dict', checkpoint)

                    is_nuscenes = isinstance(expert, NuScenesExpert)
                    if is_nuscenes:
                        remapped = {}
                        for k, v in state_dict.items():
                            if k.startswith('mlp.'):
                                remapped['decoder.' + k[len('mlp.'):]] = v
                            elif k.startswith('box_head.'):
                                remapped['bbox_head.' + k[len('box_head.'):]] = v
                            else:
                                remapped[k] = v
                        state_dict = remapped
                        expert.load_state_dict(state_dict, strict=False)
                    else:
                        expert.load_state_dict(state_dict)
                    print(f"Loaded checkpoint for expert {i}: {checkpoint_path}")
                except Exception as e:
                    warnings.warn(f"Failed to load checkpoint for expert {i}: {str(e)}")
    
    def freeze_experts(self):
        """Freeze expert parameters during gating network training"""
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
    
    def unfreeze_experts(self):
        """Unfreeze expert parameters for joint training"""
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = True


def create_automoe_model(config: Dict, device: str = 'cuda') -> AutoMoE:
    """
    Create AutoMoE model from configuration
    
    Args:
        config: Dict containing model configuration
        device: Device to place model on
    Returns:
        AutoMoE model instance
    """
    return AutoMoE(
        expert_configs=config['experts'],
        gating_config=config['gating'],
        context_config=config['context'],
        policy_config=config['policy'],
        device=device
    )

