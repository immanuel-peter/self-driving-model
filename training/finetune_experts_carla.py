"""
Fine-tuning script for expert models on CARLA data.
This implements Step 2 of the AutoMoE roadmap: Fine-tune experts on CARLA data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json

# Import expert models
from models.experts import (
    BDDDetectionExpert, 
    BDDSegmentationExpert, 
    BDDDrivableExpert, 
    NuScenesExpert
)

# Import dataloaders
import dataloaders


class ExpertFineTuner:
    def __init__(self, expert_name, model, carla_train_loader, carla_val_loader,
                 device, save_dir, learning_rate=1e-4, num_epochs=20):
        self.expert_name = expert_name
        self.model = model.to(device)
        self.carla_train_loader = carla_train_loader
        self.carla_val_loader = carla_val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer with lower learning rate for fine-tuning
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # Loss functions
        self.setup_loss_functions()
        
        # Logging
        self.writer = SummaryWriter(f'runs/{expert_name}_carla_finetune')
        self.num_epochs = num_epochs
        self.best_val_loss = float('inf')
        
    def setup_loss_functions(self):
        """Setup appropriate loss functions for each expert type"""
        if self.expert_name == 'bdd_detection':
            self.class_loss_fn = nn.CrossEntropyLoss()
            self.bbox_loss_fn = nn.SmoothL1Loss()
        elif self.expert_name in ['bdd_segmentation', 'bdd_drivable']:
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.expert_name == 'nuscenes':
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.MSELoss()
    
    def load_pretrained_weights(self, checkpoint_path):
        """Load pretrained weights from original dataset training"""
        print(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained weights loaded successfully!")
    
    def adapt_carla_sample(self, sample):
        """
        Adapt CARLA sample format to match expert's expected input format.
        CARLA samples contain different data than original datasets.
        """
        if self.expert_name == 'nuscenes':
            # NuScenes expert expects image + lidar
            if 'image' in sample and 'lidar' in sample:
                return sample
            else:
                # If no lidar, create dummy lidar data
                dummy_lidar = [torch.zeros(100, 3)]  # 100 dummy points
                return {
                    'image': sample.get('image', sample.get('rgb')),
                    'lidar': dummy_lidar
                }
        else:
            # BDD experts expect just images
            image_key = 'image' if 'image' in sample else 'rgb'
            return {
                'image': sample[image_key],
                'targets': self.create_dummy_targets(sample)
            }
    
    def create_dummy_targets(self, sample):
        """Create dummy targets for CARLA data (you'll need to adapt this)"""
        # This is a placeholder - you'll need to extract real targets from CARLA
        # For now, create dummy targets based on expert type
        if self.expert_name == 'bdd_detection':
            # Detection targets (dummy)
            return {
                'class_labels': torch.randint(0, 10, (32, 224, 224)),  # Dummy class labels
                'bbox_targets': torch.randn(32, 4, 224, 224)  # Dummy bbox targets
            }
        else:
            # Segmentation/other targets (dummy)
            return torch.randint(0, 5, (32, 224, 224))  # Dummy segmentation
    
    def compute_loss(self, outputs, sample):
        """Compute loss for CARLA fine-tuning"""
        if self.expert_name == 'bdd_detection':
            # For detection expert
            if isinstance(outputs, dict):
                # Create targets from CARLA sample (you'll need to adapt this)
                targets = self.create_dummy_targets(sample)
                class_loss = self.class_loss_fn(outputs['class_logits'], targets['class_labels'])
                bbox_loss = self.bbox_loss_fn(outputs['bbox_deltas'], targets['bbox_targets'])
                return class_loss + bbox_loss
            else:
                return torch.tensor(0.0, requires_grad=True, device=self.device)
        elif self.expert_name == 'nuscenes':
            # For NuScenes expert - you might want to predict steering/throttle from CARLA
            if 'steering' in sample and 'throttle' in sample:
                control_target = torch.stack([
                    sample['steering'], sample['throttle']
                ], dim=1)  # [B, 2]
                # Adapt output size if needed
                if outputs.size(1) != 2:
                    outputs = outputs[:, :2]  # Take first 2 outputs
                return self.loss_fn(outputs, control_target)
            else:
                return torch.tensor(0.0, requires_grad=True, device=self.device)
        else:
            # For segmentation experts
            targets = self.create_dummy_targets(sample)
            return self.loss_fn(outputs, targets)
    
    def train_epoch(self, epoch):
        """Train for one epoch on CARLA data"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.carla_train_loader, desc=f'Fine-tune Epoch {epoch+1}/{self.num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            # CARLA loader returns list of samples
            if isinstance(batch, list):
                batch = self.collate_carla_batch(batch)
            
            # Move to device
            batch = self.move_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                if self.expert_name == 'nuscenes':
                    outputs = self.model(batch)
                else:
                    images = batch.get('image', batch.get('rgb'))
                    outputs = self.model(images)
                
                loss = self.compute_loss(outputs, batch)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Log to tensorboard
                global_step = epoch * len(self.carla_train_loader) + batch_idx
                self.writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch on CARLA data"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.carla_val_loader, desc='Validation'):
                if isinstance(batch, list):
                    batch = self.collate_carla_batch(batch)
                
                batch = self.move_to_device(batch)
                
                try:
                    if self.expert_name == 'nuscenes':
                        outputs = self.model(batch)
                    else:
                        images = batch.get('image', batch.get('rgb'))
                        outputs = self.model(images)
                    
                    loss = self.compute_loss(outputs, batch)
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.writer.add_scalar('Loss/Val_Epoch', avg_loss, epoch)
        return avg_loss
    
    def collate_carla_batch(self, batch_list):
        """Collate CARLA samples into batch format"""
        if not batch_list:
            return {}
        
        # Adapt each sample first
        adapted_samples = [self.adapt_carla_sample(sample) for sample in batch_list]
        
        # Get keys from first sample
        keys = adapted_samples[0].keys()
        collated = {}
        
        for key in keys:
            values = [sample[key] for sample in adapted_samples]
            if isinstance(values[0], torch.Tensor):
                try:
                    collated[key] = torch.stack(values)
                except:
                    # If stacking fails, keep as list (e.g., for lidar)
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
        """Save fine-tuned model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'expert_name': self.expert_name,
            'finetuned_on_carla': True
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_carla_finetuned.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_carla_finetuned.pth')
            print(f"New best fine-tuned model saved with val_loss: {val_loss:.4f}")
    
    def finetune(self):
        """Main fine-tuning loop"""
        print(f"Starting CARLA fine-tuning for {self.expert_name}")
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
        
        print(f"Fine-tuning completed! Best val_loss: {self.best_val_loss:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Fine-tune expert models on CARLA')
    parser.add_argument('--expert', type=str, required=True,
                       choices=['bdd_detection', 'bdd_segmentation', 'bdd_drivable', 'nuscenes'],
                       help='Which expert to fine-tune')
    parser.add_argument('--pretrained_path', type=str, required=True,
                       help='Path to pretrained expert checkpoint')
    parser.add_argument('--epochs', type=int, default=20, help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Fine-tuning learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir) / f"{args.expert}_carla_finetuned"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save fine-tuning config
    with open(save_dir / 'finetune_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup model
    if args.expert == 'bdd_detection':
        model = BDDDetectionExpert(num_classes=10)
    elif args.expert == 'bdd_segmentation':
        model = BDDSegmentationExpert()
    elif args.expert == 'bdd_drivable':
        model = BDDDrivableExpert()
    elif args.expert == 'nuscenes':
        model = NuScenesExpert()
    
    # Setup CARLA dataloaders
    carla_train_loader = dataloaders.carla_train_loader
    carla_val_loader = dataloaders.carla_val_loader
    
    # Create fine-tuner
    finetuner = ExpertFineTuner(
        expert_name=args.expert,
        model=model,
        carla_train_loader=carla_train_loader,
        carla_val_loader=carla_val_loader,
        device=device,
        save_dir=save_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )
    
    # Load pretrained weights
    finetuner.load_pretrained_weights(args.pretrained_path)
    
    # Start fine-tuning
    finetuner.finetune()


if __name__ == '__main__':
    main()
