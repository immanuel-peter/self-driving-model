import torch
import torch.nn as nn
import torchvision.models as models


class CarlaExpert(nn.Module):
    """
    CARLA-specific expert model for handling simulation-specific patterns.
    This expert is designed to handle CARLA's unique visual characteristics
    and control paradigms.
    """
    
    def __init__(self, 
                 output_dim=2,  # steering + throttle
                 backbone='resnet18',
                 use_auxiliary_tasks=True,
                 pretrained_backbone=True):
        super().__init__()
        
        self.output_dim = output_dim
        self.use_auxiliary_tasks = use_auxiliary_tasks
        
        # Main backbone for feature extraction
        self.setup_backbone(backbone, pretrained_backbone)
        
        # Main control head
        self.control_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
        
        # Auxiliary tasks for better representation learning
        if use_auxiliary_tasks:
            self.setup_auxiliary_heads()
    
    def setup_backbone(self, backbone, pretrained):
        """Setup the CNN backbone"""
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.backbone_dim = 512
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.backbone_dim = 512
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Additional adaptation layers for CARLA-specific features
        self.adaptation_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    
    def setup_auxiliary_heads(self):
        """Setup auxiliary task heads for better learning"""
        # Speed prediction head
        self.speed_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Traffic light state prediction
        self.traffic_light_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Red, Yellow, Green
        )
        
        # Waypoint prediction (relative position of next waypoint)
        self.waypoint_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # x, y offset
        )
        
        # Hazard detection (binary classification)
        self.hazard_head = nn.Sequential(
            nn.Linear(self.backbone_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image tensor [B, 3, H, W]
            
        Returns:
            Dictionary with main output and auxiliary predictions
        """
        # Extract features
        features = self.backbone(x)  # [B, backbone_dim, 1, 1]
        features = features.view(features.size(0), -1)  # [B, backbone_dim]
        
        # Main control prediction
        control_output = self.control_head(features)  # [B, output_dim]
        
        output = {
            'control': control_output,
            'features': features
        }
        
        # Auxiliary predictions
        if self.use_auxiliary_tasks:
            output.update({
                'speed': self.speed_head(features),
                'traffic_light': self.traffic_light_head(features),
                'waypoint': self.waypoint_head(features),
                'hazard': self.hazard_head(features)
            })
        
        return output
    
    def predict(self, x):
        """Inference function with post-processing"""
        output = self.forward(x)
        
        # Apply appropriate activations for control outputs
        control = output['control']
        
        # Steering: tanh to constrain to [-1, 1]
        steering = torch.tanh(control[:, 0])
        
        # Throttle: sigmoid to constrain to [0, 1]
        throttle = torch.sigmoid(control[:, 1])
        
        processed_output = {
            'steering': steering,
            'throttle': throttle,
            'control': torch.stack([steering, throttle], dim=1)
        }
        
        # Add auxiliary predictions with activations
        if self.use_auxiliary_tasks:
            processed_output.update({
                'speed': torch.relu(output['speed']),  # Speed should be positive
                'traffic_light': torch.softmax(output['traffic_light'], dim=1),
                'waypoint': output['waypoint'],  # Relative position can be negative
                'hazard': torch.sigmoid(output['hazard'])  # Probability
            })
        
        return processed_output
    
    def compute_auxiliary_losses(self, predictions, targets):
        """
        Compute auxiliary task losses
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets dictionary
            
        Returns:
            Dictionary of auxiliary losses
        """
        if not self.use_auxiliary_tasks:
            return {}
        
        losses = {}
        
        # Speed loss
        if 'speed' in targets and 'speed' in predictions:
            losses['speed'] = nn.MSELoss()(predictions['speed'].squeeze(), targets['speed'])
        
        # Traffic light loss
        if 'traffic_light' in targets and 'traffic_light' in predictions:
            losses['traffic_light'] = nn.CrossEntropyLoss()(
                predictions['traffic_light'], targets['traffic_light']
            )
        
        # Waypoint loss
        if 'waypoint' in targets and 'waypoint' in predictions:
            losses['waypoint'] = nn.MSELoss()(predictions['waypoint'], targets['waypoint'])
        
        # Hazard loss
        if 'hazard' in targets and 'hazard' in predictions:
            losses['hazard'] = nn.BCELoss()(
                predictions['hazard'].squeeze(), targets['hazard'].float()
            )
        
        return losses
    
    def compute_total_loss(self, predictions, targets, aux_loss_weights=None):
        """
        Compute total loss including main task and auxiliary tasks
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            aux_loss_weights: Weights for auxiliary losses
        """
        if aux_loss_weights is None:
            aux_loss_weights = {
                'speed': 0.1,
                'traffic_light': 0.1,
                'waypoint': 0.2,
                'hazard': 0.1
            }
        
        # Main control loss
        if 'control' in targets:
            main_loss = nn.MSELoss()(predictions['control'], targets['control'])
        else:
            # If no control target, create dummy loss
            main_loss = torch.tensor(0.0, requires_grad=True, device=predictions['control'].device)
        
        total_loss = main_loss
        loss_components = {'main': main_loss.item()}
        
        # Auxiliary losses
        aux_losses = self.compute_auxiliary_losses(predictions, targets)
        
        for task, loss in aux_losses.items():
            weight = aux_loss_weights.get(task, 0.1)
            total_loss += weight * loss
            loss_components[f'aux_{task}'] = loss.item()
        
        return total_loss, loss_components
