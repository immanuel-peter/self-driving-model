"""
Training script for the gating network.
This implements Step 4 of the AutoMoE roadmap: Train gating mechanism.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# Import models
from models.gating_network import GatingNetwork, MoEModel
from models.experts import (
    BDDDetectionExpert, 
    BDDSegmentationExpert, 
    BDDDrivableExpert, 
    NuScenesExpert
)

# Import dataloaders
import dataloaders


class GatingTrainer:
    def __init__(self, 
                 moe_model: MoEModel,
                 carla_train_loader,
                 carla_val_loader,
                 device,
                 save_dir,
                 learning_rate=1e-3,
                 num_epochs=30,
                 oracle_labeling=True):
        
        self.moe_model = moe_model.to(device)
        self.carla_train_loader = carla_train_loader
        self.carla_val_loader = carla_val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.oracle_labeling = oracle_labeling
        self.num_epochs = num_epochs
        
        # Only train gating network parameters (experts are frozen)
        gating_params = list(self.moe_model.gating_network.parameters())
        projection_params = list(self.moe_model.expert_projections.parameters())
        
        self.optimizer = optim.Adam(
            gating_params + projection_params, 
            lr=learning_rate, 
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Loss functions
        self.imitation_loss_fn = nn.MSELoss()
        self.oracle_loss_fn = nn.CrossEntropyLoss()
        
        # Logging
        self.writer = SummaryWriter(f'runs/gating_network')
        self.best_val_loss = float('inf')
        
    def load_expert_checkpoints(self, expert_checkpoint_paths):
        """Load pretrained/fine-tuned expert checkpoints"""
        print("Loading expert checkpoints...")
        
        for expert_name, checkpoint_path in expert_checkpoint_paths.items():
            if expert_name in self.moe_model.experts:
                print(f"Loading {expert_name} from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.moe_model.experts[expert_name].load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ {expert_name} loaded successfully")
        
        # Freeze expert parameters
        for expert in self.moe_model.experts.values():
            for param in expert.parameters():
                param.requires_grad = False
    
    def compute_oracle_labels(self, sample):
        """
        Compute oracle labels: which expert performs best on this sample.
        This is used for supervised gating training.
        """
        with torch.no_grad():
            image = sample.get('image', sample.get('rgb'))
            target_control = self.extract_control_target(sample)
            
            if target_control is None:
                # If no control target, return random expert
                return torch.randint(0, len(self.moe_model.experts), (image.size(0),), device=self.device)
            
            # Get predictions from all experts
            expert_errors = {}
            
            for expert_name, expert_model in self.moe_model.experts.items():
                try:
                    if expert_name == 'nuscenes':
                        # NuScenes expert needs image + lidar
                        nuscenes_input = {
                            'image': image,
                            'lidar': sample.get('lidar', [torch.zeros(100, 3, device=self.device) for _ in range(image.size(0))])
                        }
                        expert_output = expert_model(nuscenes_input)
                    else:
                        # BDD experts use image only
                        expert_output = expert_model(image)
                    
                    # Project to control space
                    if isinstance(expert_output, dict):
                        # Handle detection output
                        expert_features = torch.cat([
                            expert_output['class_logits'].mean(dim=[2, 3]),
                            expert_output['bbox_deltas'].mean(dim=[2, 3])
                        ], dim=1)
                    elif len(expert_output.shape) == 4:
                        # Handle segmentation output [B, C, H, W]
                        expert_features = expert_output.mean(dim=[2, 3])
                    else:
                        expert_features = expert_output
                    
                    # Project to control output
                    control_pred = self.moe_model.expert_projections[expert_name](expert_features)
                    
                    # Compute error
                    error = self.imitation_loss_fn(control_pred, target_control)
                    expert_errors[expert_name] = error.item()
                    
                except Exception as e:
                    print(f"Error evaluating {expert_name}: {e}")
                    expert_errors[expert_name] = float('inf')
            
            # Find best expert for each sample
            expert_names = list(self.moe_model.experts.keys())
            oracle_labels = []
            
            for b in range(image.size(0)):
                # For this batch sample, find expert with lowest error
                best_expert_idx = 0
                best_error = float('inf')
                
                for i, expert_name in enumerate(expert_names):
                    if expert_errors[expert_name] < best_error:
                        best_error = expert_errors[expert_name]
                        best_expert_idx = i
                
                oracle_labels.append(best_expert_idx)
            
            return torch.tensor(oracle_labels, device=self.device)
    
    def extract_control_target(self, sample):
        """Extract control target (steering, throttle) from CARLA sample"""
        if 'steering' in sample and 'throttle' in sample:
            steering = sample['steering']
            throttle = sample['throttle']
            
            # Handle different input formats
            if isinstance(steering, (list, tuple)):
                steering = torch.tensor(steering, device=self.device)
            if isinstance(throttle, (list, tuple)):
                throttle = torch.tensor(throttle, device=self.device)
            
            # Ensure proper shapes
            if len(steering.shape) == 0:
                steering = steering.unsqueeze(0)
            if len(throttle.shape) == 0:
                throttle = throttle.unsqueeze(0)
            
            control_target = torch.stack([steering, throttle], dim=1)  # [B, 2]
            return control_target
        else:
            return None
    
    def extract_context_data(self, sample):
        """Extract weather and traffic context from CARLA sample"""
        weather_data = {}
        traffic_data = {}
        
        # Extract weather context
        if 'weather' in sample:
            weather = sample['weather']
            weather_data = {
                'category': weather.get('category', torch.zeros(1, dtype=torch.long, device=self.device)),
                'visibility': weather.get('visibility', torch.ones(1, device=self.device)),
                'wetness': weather.get('wetness', torch.zeros(1, device=self.device)),
                'cloudiness': weather.get('cloudiness', torch.zeros(1, device=self.device))
            }
        
        # Extract traffic context
        if 'traffic' in sample:
            traffic = sample['traffic']
            traffic_data = {
                'vehicle_count': traffic.get('vehicle_count', torch.zeros(1, device=self.device)),
                'pedestrian_count': traffic.get('pedestrian_count', torch.zeros(1, device=self.device)),
                'traffic_light_state': traffic.get('traffic_light_state', torch.zeros(1, device=self.device)),
                'speed_limit': traffic.get('speed_limit', torch.ones(1, device=self.device) * 50.0)
            }
        
        return weather_data if weather_data else None, traffic_data if traffic_data else None
    
    def compute_loss(self, moe_output, sample):
        """Compute training loss for gating network"""
        if self.oracle_labeling:
            # Supervised learning: predict best expert
            oracle_labels = self.compute_oracle_labels(sample)
            expert_logits = moe_output['expert_weights']
            oracle_loss = self.oracle_loss_fn(expert_logits, oracle_labels)
            
            # Also include imitation loss
            control_target = self.extract_control_target(sample)
            if control_target is not None:
                imitation_loss = self.imitation_loss_fn(moe_output['final_output'], control_target)
                total_loss = oracle_loss + 0.5 * imitation_loss
                return total_loss, oracle_loss.item(), imitation_loss.item()
            else:
                return oracle_loss, oracle_loss.item(), 0.0
        else:
            # End-to-end imitation learning
            control_target = self.extract_control_target(sample)
            if control_target is not None:
                imitation_loss = self.imitation_loss_fn(moe_output['final_output'], control_target)
                return imitation_loss, 0.0, imitation_loss.item()
            else:
                # Return dummy loss
                return torch.tensor(0.0, requires_grad=True, device=self.device), 0.0, 0.0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.moe_model.train()
        total_loss = 0
        total_oracle_loss = 0
        total_imitation_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.carla_train_loader, desc=f'Gating Epoch {epoch+1}/{self.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Handle batch format
            if isinstance(batch, list):
                batch = self.collate_carla_batch(batch)
            
            batch = self.move_to_device(batch)
            
            try:
                # Extract inputs
                image = batch.get('image', batch.get('rgb'))
                lidar = batch.get('lidar', None)
                weather_data, traffic_data = self.extract_context_data(batch)
                
                # Forward pass
                self.optimizer.zero_grad()
                moe_output = self.moe_model(image, lidar, weather_data, traffic_data)
                
                # Compute loss
                loss, oracle_loss, imitation_loss = self.compute_loss(moe_output, batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Accumulate losses
                total_loss += loss.item()
                total_oracle_loss += oracle_loss
                total_imitation_loss += imitation_loss
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'oracle': f'{oracle_loss:.4f}',
                    'imitation': f'{imitation_loss:.4f}'
                })
                
                # Log to tensorboard
                global_step = epoch * len(self.carla_train_loader) + batch_idx
                self.writer.add_scalar('Loss/Train_Total', loss.item(), global_step)
                self.writer.add_scalar('Loss/Train_Oracle', oracle_loss, global_step)
                self.writer.add_scalar('Loss/Train_Imitation', imitation_loss, global_step)
                
                # Log expert selection statistics
                if batch_idx % 100 == 0:
                    expert_stats = self.moe_model.gating_network.get_expert_selection_stats(
                        image, weather_data, traffic_data
                    )
                    for stat_name, stat_value in expert_stats.items():
                        self.writer.add_scalar(f'Expert_Stats/{stat_name}', stat_value, global_step)
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Average losses
        avg_loss = total_loss / max(num_batches, 1)
        avg_oracle_loss = total_oracle_loss / max(num_batches, 1)
        avg_imitation_loss = total_imitation_loss / max(num_batches, 1)
        
        self.writer.add_scalar('Loss/Train_Epoch_Total', avg_loss, epoch)
        self.writer.add_scalar('Loss/Train_Epoch_Oracle', avg_oracle_loss, epoch)
        self.writer.add_scalar('Loss/Train_Epoch_Imitation', avg_imitation_loss, epoch)
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.moe_model.eval()
        total_loss = 0
        total_oracle_loss = 0
        total_imitation_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.carla_val_loader, desc='Validation'):
                if isinstance(batch, list):
                    batch = self.collate_carla_batch(batch)
                
                batch = self.move_to_device(batch)
                
                try:
                    # Extract inputs
                    image = batch.get('image', batch.get('rgb'))
                    lidar = batch.get('lidar', None)
                    weather_data, traffic_data = self.extract_context_data(batch)
                    
                    # Forward pass
                    moe_output = self.moe_model(image, lidar, weather_data, traffic_data)
                    
                    # Compute loss
                    loss, oracle_loss, imitation_loss = self.compute_loss(moe_output, batch)
                    
                    total_loss += loss.item()
                    total_oracle_loss += oracle_loss
                    total_imitation_loss += imitation_loss
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        # Average losses
        avg_loss = total_loss / max(num_batches, 1)
        avg_oracle_loss = total_oracle_loss / max(num_batches, 1)
        avg_imitation_loss = total_imitation_loss / max(num_batches, 1)
        
        self.writer.add_scalar('Loss/Val_Epoch_Total', avg_loss, epoch)
        self.writer.add_scalar('Loss/Val_Epoch_Oracle', avg_oracle_loss, epoch)
        self.writer.add_scalar('Loss/Val_Epoch_Imitation', avg_imitation_loss, epoch)
        
        return avg_loss
    
    def collate_carla_batch(self, batch_list):
        """Collate CARLA samples into batch format"""
        if not batch_list:
            return {}
        
        keys = batch_list[0].keys()
        collated = {}
        
        for key in keys:
            values = [sample[key] for sample in batch_list]
            if isinstance(values[0], torch.Tensor):
                try:
                    collated[key] = torch.stack(values)
                except:
                    collated[key] = values
            else:
                collated[key] = values
        
        return collated
    
    def move_to_device(self, batch):
        """Move batch to device"""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save gating network checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'gating_network_state_dict': self.moe_model.gating_network.state_dict(),
            'expert_projections_state_dict': self.moe_model.expert_projections.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'oracle_labeling': self.oracle_labeling
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_gating.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_gating.pth')
            print(f"New best gating network saved with val_loss: {val_loss:.4f}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting gating network training")
        print(f"Oracle labeling: {self.oracle_labeling}")
        print(f"CARLA training samples: {len(self.carla_train_loader.dataset)}")
        print(f"CARLA validation samples: {len(self.carla_val_loader.dataset)}")
        
        for epoch in range(self.num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            print(f"Epoch {epoch+1}/{self.num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        print(f"Gating network training completed! Best val_loss: {self.best_val_loss:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train gating network for MoE model')
    parser.add_argument('--expert_checkpoints', type=str, required=True,
                       help='JSON file with expert checkpoint paths')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='checkpoints/gating', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--oracle_labeling', action='store_true', help='Use oracle labeling for gating training')
    parser.add_argument('--gating_type', type=str, default='soft', choices=['soft', 'hard'], help='Gating type')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'gating_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load expert checkpoint paths
    with open(args.expert_checkpoints, 'r') as f:
        expert_checkpoint_paths = json.load(f)
    
    # Create expert models
    experts = {
        'bdd_detection': BDDDetectionExpert(num_classes=10),
        'bdd_segmentation': BDDSegmentationExpert(),
        'bdd_drivable': BDDDrivableExpert(),
        'nuscenes': NuScenesExpert()
    }
    
    # Create gating network
    gating_network = GatingNetwork(
        num_experts=len(experts),
        gating_type=args.gating_type,
        use_weather_context=True,
        use_traffic_context=True
    )
    
    # Create MoE model
    moe_model = MoEModel(
        experts=experts,
        gating_network=gating_network,
        output_dim=2,  # steering + throttle
        freeze_experts=True
    )
    
    # Setup CARLA dataloaders
    carla_train_loader = dataloaders.carla_train_loader
    carla_val_loader = dataloaders.carla_val_loader
    
    # Create trainer
    trainer = GatingTrainer(
        moe_model=moe_model,
        carla_train_loader=carla_train_loader,
        carla_val_loader=carla_val_loader,
        device=device,
        save_dir=save_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        oracle_labeling=args.oracle_labeling
    )
    
    # Load expert checkpoints
    trainer.load_expert_checkpoints(expert_checkpoint_paths)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
