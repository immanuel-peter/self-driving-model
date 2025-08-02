"""
Training script for individual expert models on their original datasets.
This implements Step 1 of the AutoMoE roadmap: Train experts on large-scale data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
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


class ExpertTrainer:
    def __init__(self, expert_name, model, train_loader, val_loader, 
                 device, save_dir, learning_rate=1e-3, num_epochs=50):
        self.expert_name = expert_name
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer and loss functions
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Loss functions based on expert type
        self.setup_loss_functions()
        
        # Logging
        self.writer = SummaryWriter(f'runs/{expert_name}')
        self.num_epochs = num_epochs
        self.best_val_loss = float('inf')
        
    def setup_loss_functions(self):
        """Setup appropriate loss functions for each expert type"""
        if self.expert_name == 'bdd_detection':
            # Detection: classification + regression loss
            self.class_loss_fn = nn.CrossEntropyLoss()
            self.bbox_loss_fn = nn.SmoothL1Loss()
        elif self.expert_name in ['bdd_segmentation', 'bdd_drivable']:
            # Segmentation: cross-entropy loss
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.expert_name == 'nuscenes':
            # Multimodal regression/classification
            self.loss_fn = nn.MSELoss()  # Adjust based on your target
        else:
            # Default regression loss
            self.loss_fn = nn.MSELoss()
    
    def compute_loss(self, outputs, targets):
        """Compute loss based on expert type"""
        if self.expert_name == 'bdd_detection':
            # Detection expert has classification and bbox regression
            if isinstance(outputs, dict):
                class_loss = self.class_loss_fn(outputs['class_logits'], targets['class_labels'])
                bbox_loss = self.bbox_loss_fn(outputs['bbox_deltas'], targets['bbox_targets'])
                return class_loss + bbox_loss
            else:
                # If outputs is not dict, assume it's for generic training
                return self.loss_fn(outputs, targets)
        else:
            return self.loss_fn(outputs, targets)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if isinstance(batch, list):
                # CARLA/NuScenes format - list of samples
                batch = self.collate_batch(batch)
            
            # Move to device
            batch = self.move_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.expert_name == 'nuscenes':
                outputs = self.model(batch)
                # For NuScenes, assume we have some target in the batch
                targets = batch.get('targets', torch.randn_like(outputs))  # Placeholder
            else:
                # For BDD experts, assume image input
                images = batch.get('image', batch.get('images'))
                outputs = self.model(images)
                targets = batch.get('targets', torch.randn_like(outputs))  # Placeholder
            
            loss = self.compute_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Handle different batch formats
                if isinstance(batch, list):
                    batch = self.collate_batch(batch)
                
                # Move to device
                batch = self.move_to_device(batch)
                
                # Forward pass
                if self.expert_name == 'nuscenes':
                    outputs = self.model(batch)
                    targets = batch.get('targets', torch.randn_like(outputs))  # Placeholder
                else:
                    images = batch.get('image', batch.get('images'))
                    outputs = self.model(images)
                    targets = batch.get('targets', torch.randn_like(outputs))  # Placeholder
                
                loss = self.compute_loss(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/Val_Epoch', avg_loss, epoch)
        return avg_loss
    
    def collate_batch(self, batch_list):
        """Collate list of samples into a batch"""
        # Simple collation - you may need to customize this
        if not batch_list:
            return {}
        
        # Get keys from first sample
        keys = batch_list[0].keys()
        collated = {}
        
        for key in keys:
            values = [sample[key] for sample in batch_list]
            if isinstance(values[0], torch.Tensor):
                try:
                    collated[key] = torch.stack(values)
                except:
                    # If stacking fails, keep as list
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
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'expert_name': self.expert_name
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pth')
            print(f"New best model saved with val_loss: {val_loss:.4f}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.expert_name}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
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
        
        print(f"Training completed! Best val_loss: {self.best_val_loss:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train expert models')
    parser.add_argument('--expert', type=str, required=True,
                       choices=['bdd_detection', 'bdd_segmentation', 'bdd_drivable', 'nuscenes'],
                       help='Which expert to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.expert
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup model and dataloaders based on expert type
    if args.expert == 'bdd_detection':
        model = BDDDetectionExpert(num_classes=10)
        train_loader = dataloaders.bdd_detection_train_loader
        val_loader = dataloaders.bdd_detection_val_loader
    elif args.expert == 'bdd_segmentation':
        model = BDDSegmentationExpert()
        train_loader = dataloaders.bdd_segmentation_train_loader
        val_loader = dataloaders.bdd_segmentation_val_loader
    elif args.expert == 'bdd_drivable':
        model = BDDDrivableExpert()
        train_loader = dataloaders.bdd_drivable_train_loader
        val_loader = dataloaders.bdd_drivable_val_loader
    elif args.expert == 'nuscenes':
        model = NuScenesExpert()
        train_loader = dataloaders.nuscenes_train_loader
        val_loader = dataloaders.nuscenes_val_loader
    
    # Create trainer and start training
    trainer = ExpertTrainer(
        expert_name=args.expert,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
