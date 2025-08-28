import argparse
import json
import os
import sys
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.automoe import create_automoe_model
from dataloaders.carla_sequence_loader import CarlaSequenceDataset, carla_sequence_collate

def compute_gating_losses(pred: Dict[str, torch.Tensor], 
                         target_wp: torch.Tensor, 
                         target_spd: torch.Tensor,
                         config: Dict) -> Dict[str, torch.Tensor]:
    """Compute losses for gating network training"""
    losses = {}
    
    # Policy losses
    ade = F.l1_loss(pred["waypoints"], target_wp)
    fde = F.l1_loss(pred["waypoints"][:, -1, :], target_wp[:, -1, :])
    # Prefer full sequence if available
    pred_spd = pred.get("speed_seq", pred.get("speed"))
    if pred_spd is not None and pred_spd.dim() == 2 and target_spd.dim() == 2 and pred_spd.size(1) == target_spd.size(1):
        speed_loss = F.l1_loss(pred_spd, target_spd)
    else:
        # Fall back to last-step speed if shapes differ
        pred_last = pred.get("speed")
        if pred_last is not None and pred_last.dim() == 2 and pred_last.size(1) == 1:
            target_last = target_spd[:, -1:].contiguous()
            speed_loss = F.l1_loss(pred_last, target_last)
        else:
            # If still incompatible, use zeros to avoid crash
            speed_loss = torch.zeros((), device=target_spd.device)
    
    # Smoothness loss on waypoints
    pred_deltas = pred["waypoints"][:, 1:, :] - pred["waypoints"][:, :-1, :]
    smoothness_loss = F.l1_loss(pred_deltas[:, 1:, :], pred_deltas[:, :-1, :])
    
    # Gating regularization losses
    expert_weights = pred["expert_weights"]  # [B, num_experts]
    
    # Load balancing loss (encourage equal expert usage)
    if config.get('use_load_balancing', True):
        mean_usage = expert_weights.mean(dim=0)  # [num_experts]
        target_usage = torch.ones_like(mean_usage) / mean_usage.size(0)
        load_balancing_loss = F.mse_loss(mean_usage, target_usage)
    else:
        load_balancing_loss = torch.tensor(0.0, device=expert_weights.device)
    
    # Entropy loss (encourage confident expert selection)
    if config.get('use_entropy_loss', True):
        entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum(dim=1).mean()
        entropy_loss = -entropy  # Negative entropy to encourage confident selection
    else:
        entropy_loss = torch.tensor(0.0, device=expert_weights.device)
    
    # Combine losses
    total_loss = (
        config.get('ade_weight', 1.0) * ade +
        config.get('fde_weight', 2.0) * fde +
        config.get('speed_weight', 0.2) * speed_loss +
        config.get('smoothness_weight', 0.1) * smoothness_loss +
        config.get('load_balancing_weight', 0.01) * load_balancing_loss +
        config.get('entropy_weight', 0.001) * entropy_loss
    )
    
    return {
        "total_loss": total_loss,
        "ade": ade,
        "fde": fde,
        "speed": speed_loss,
        "smoothness": smoothness_loss,
        "load_balancing": load_balancing_loss,
        "entropy": entropy_loss
    }

def train_one_epoch(model: nn.Module, 
                   loader: DataLoader, 
                   optimizer: optim.Optimizer,
                   device: torch.device, 
                   epoch_idx: int, 
                   epochs: int, 
                   rank: int,
                   config: Dict) -> float:
    """Train for one epoch"""
    model.train()
    desc = f"train | epoch {epoch_idx+1}/{epochs}"
    it = loader if rank != 0 else tqdm(loader, desc=desc)
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in it:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(batch)
        
        # Compute losses
        losses = compute_gating_losses(pred, batch["waypoints"], batch["speed"], config)
        
        # Backward pass
        losses["total_loss"].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += float(losses["total_loss"].item())
        num_batches += 1
        
        if rank == 0:
            it.set_postfix({
                "loss": f"{losses['total_loss'].item():.3f}",
                "ade": f"{losses['ade'].item():.3f}",
                "fde": f"{losses['fde'].item():.3f}"
            })
    
    return total_loss / max(1, num_batches)

@torch.no_grad()
def validate(model: nn.Module, 
             loader: DataLoader, 
             device: torch.device, 
             epoch_idx: int, 
             epochs: int, 
             rank: int, 
             world_size: int,
             config: Dict) -> float:
    """Validate model"""
    model.eval()
    desc = f"val | epoch {epoch_idx+1}/{epochs}"
    it = loader if rank != 0 else tqdm(loader, desc=desc)
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in it:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        pred = model(batch)
        
        # Compute losses
        losses = compute_gating_losses(pred, batch["waypoints"], batch["speed"], config)
        
        total_loss += float(losses["total_loss"].item())
        num_batches += 1
        
        if rank == 0:
            it.set_postfix({
                "loss": f"{losses['total_loss'].item():.3f}",
                "ade": f"{losses['ade'].item():.3f}",
                "fde": f"{losses['fde'].item():.3f}"
            })
    
    # Reduce across ranks
    if dist.is_initialized():
        t = torch.tensor([total_loss, num_batches], dtype=torch.float32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, num_batches = float(t[0].item()), int(t[1].item())
    
    return total_loss / max(1, num_batches)

def save_checkpoint(model: nn.Module, 
                   optimizer: optim.Optimizer, 
                   epoch: int, 
                   loss: float, 
                   save_path: str,
                   rank: int):
    """Save model checkpoint"""
    if rank == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")

def load_checkpoint(model: nn.Module, 
                   optimizer: optim.Optimizer, 
                   checkpoint_path: str,
                   device: torch.device) -> int:
    """Load model checkpoint"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from epoch 0")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Train AutoMoE gating network")
    parser.add_argument("--config", type=str, required=True, help="Path to training config file")
    parser.add_argument("--model_config", type=str, default="models/configs/automoe/model_config.json", help="Path to model config file")
    parser.add_argument("--data_root", type=str, required=True, help="Path to CARLA data")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints/gating", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--expert_checkpoints", nargs="+", default=[], help="Expert checkpoint paths")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for DDP")
    parser.add_argument("--world_size", type=int, default=1, help="World size for DDP")
    args = parser.parse_args()
    
    # Load configurations
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = json.load(f)
    
    # Setup DDP (use torchrun-provided env vars when available)
    env_world_size = int(os.environ.get('WORLD_SIZE', str(args.world_size)))
    if env_world_size > 1:
        local_rank = int(os.environ.get('LOCAL_RANK', str(args.local_rank)))
        world_size = int(os.environ.get('WORLD_SIZE', str(args.world_size)))
        rank = int(os.environ.get('RANK', '0'))

        # Reflect resolved ranks back to args for downstream logging
        args.local_rank = local_rank
        args.world_size = world_size

        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_automoe_model(model_config, device)
    
    # Load expert checkpoints if provided
    if args.expert_checkpoints:
        model.load_expert_checkpoints(args.expert_checkpoints)
        print("Loaded expert checkpoints")
    
    # Freeze experts during gating training
    model.freeze_experts()
    print("Frozen expert parameters")
    
    # Wrap with DDP if using multiple GPUs
    if args.world_size > 1:
        model = DDP(model, device_ids=[args.local_rank])
    
    # Create datasets
    train_dataset = CarlaSequenceDataset(
        split='train',
        root_dir=args.data_root,
        horizon=config.get('horizon', config.get('sequence_length', 10)),
        include_context=not config.get('no_context', False)
    )
    
    val_dataset = CarlaSequenceDataset(
        split='val',
        root_dir=args.data_root,
        horizon=config.get('horizon', config.get('sequence_length', 10)),
        include_context=not config.get('no_context', False)
    )
    
    # Create dataloaders
    if args.world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.get('num_workers', 4),
        collate_fn=carla_sequence_collate,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.get('num_workers', 4),
        collate_fn=carla_sequence_collate,
        pin_memory=True
    )
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.get('epochs', 100) * len(train_loader)
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup tensorboard
    if args.local_rank == 0:
        writer = SummaryWriter(f"models/runs/gating_network_{config.get('run_name', 'default')}")
    
    # Training loop
    best_val_loss = float('inf')
    epochs = config.get('epochs', 100)
    
    for epoch in range(start_epoch, epochs):
        if args.world_size > 1:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, epochs, args.local_rank, config
        )
        
        # Validate
        val_loss = validate(
            model, val_loader, device, epoch, epochs, args.local_rank, args.world_size, config
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        if args.local_rank == 0:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.checkpoint_dir, 'best.pth'),
                args.local_rank
            )
        
        # Save latest checkpoint
        if (epoch + 1) % config.get('save_freq', 10) == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth'),
                args.local_rank
            )
    
    # Cleanup
    if args.world_size > 1:
        dist.destroy_process_group()
    
    if args.local_rank == 0:
        writer.close()
        print("Training completed!")

if __name__ == "__main__":
    main()
