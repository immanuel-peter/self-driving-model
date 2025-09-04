# training/train_nuscenes_expert_ddp.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hungarian_matcher import HungarianMatcher
from models.experts import NuScenesExpert
from dataloaders.nuscenes_loader import get_nuscenes_loader

class NuScenesTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        if isinstance(model, DDP):
            model.module.to(device)
            self.model = model
        else:
            self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs'] * len(self.train_loader)
        )
        
        # detection losses: CE over 10 classes, SmoothL1 on 7-dim boxes
        self.class_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.bbox_loss_fn  = nn.SmoothL1Loss(reduction='none')
        # Hungarian matching across our 1 "query" per sample
        self.matcher = HungarianMatcher(
            cost_class=self.config.get('cost_class',1.0),
            cost_bbox=self.config.get('cost_bbox',5.0),
            cost_giou=self.config.get('cost_giou',2.0)
        )
        
        if dist.is_initialized() and dist.get_rank() == 0:
            self.writer = SummaryWriter(f"models/runs/nuscenes_expert_{config['run_name']}")
        else:
            self.writer = None
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        if dist.is_initialized() and dist.get_rank() == 0:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")
        else:
            pbar = self.train_loader
        
        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        for batch in pbar:
            self.optimizer.zero_grad()
            
            batch_dict = {
                'image': batch['image'].to(self.device),
                'lidar': batch['lidar'].to(self.device),
                'intrinsics': batch['intrinsics'].to(self.device)
            }

            outputs     = self.model(batch_dict)
            pred_logits = outputs['class_logits']  # [B,10]
            pred_boxes  = outputs['bbox_preds']    # [B,7]

            gt_boxes  = batch['boxes'].to(self.device)  # [B, M, 7]
            gt_labels = batch['labels'].to(self.device) # [B, M]
            targets = []
            B, M, _ = gt_boxes.shape
            for i in range(B):
                mask = gt_labels[i] != -1
                targets.append({
                    'boxes':  gt_boxes[i][mask],    # [Ni,7]
                    'labels': gt_labels[i][mask]    # [Ni]
                })

            # pred_logits: [B, num_queries, 10], pred_boxes: [B, num_queries, 7]
            indices = self.matcher(
                {'pred_logits': pred_logits, 'pred_boxes': pred_boxes},
                targets
            )

            # classification: pred_logits [B, num_queries, 10], targets [B, num_queries]
            B, num_queries, num_classes = pred_logits.shape
            tgt_classes = torch.full((B, num_queries), -1, dtype=torch.int64, device=self.device)
            for i, (pred_idx, tgt_idx) in enumerate(indices):
                tgt_classes[i, pred_idx] = targets[i]['labels'][tgt_idx]
            loss_cls = self.class_loss_fn(pred_logits.view(-1, num_classes), tgt_classes.view(-1))

            tgt_boxes = torch.zeros_like(pred_boxes)
            for i, (pred_idx, tgt_idx) in enumerate(indices):
                tgt_boxes[i, pred_idx] = targets[i]['boxes'][tgt_idx]
            loss_bbox = self.bbox_loss_fn(pred_boxes, tgt_boxes).mean()

            loss = loss_cls + self.config.get('bbox_loss_weight',5.0) * loss_bbox
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            if dist.is_initialized() and dist.get_rank() == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if self.writer is not None:
                step = epoch * len(self.train_loader) + (pbar.n if hasattr(pbar, 'n') else 0)
                self.writer.add_scalar('train/loss_batch', loss.item(), step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], step)
            
        avg_loss = total_loss / len(self.train_loader)
        if self.writer is not None:
            self.writer.add_scalar('train/loss_epoch', avg_loss, epoch)
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        if dist.is_initialized() and dist.get_rank() == 0:
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        else:
            pbar = self.val_loader
        
        if hasattr(self.val_loader, 'sampler') and hasattr(self.val_loader.sampler, 'set_epoch'):
            self.val_loader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for batch in pbar:
                batch_dict = {
                    'image': batch['image'].to(self.device),
                    'lidar': batch['lidar'].to(self.device),
                    'intrinsics': batch['intrinsics'].to(self.device)
                }
                outputs     = self.model(batch_dict)
                pred_logits = outputs['class_logits']  # [B,10]
                pred_boxes  = outputs['bbox_preds']    # [B,7]

                gt_boxes  = batch['boxes'].to(self.device)  # [B, M, 7]
                gt_labels = batch['labels'].to(self.device) # [B, M]
                targets = []
                B, M, _ = gt_boxes.shape
                for i in range(B):
                    mask = gt_labels[i] != -1
                    targets.append({
                        'boxes':  gt_boxes[i][mask],    # [Ni,7]
                        'labels': gt_labels[i][mask]    # [Ni]
                    })

                indices = self.matcher(
                    {'pred_logits': pred_logits, 'pred_boxes': pred_boxes},
                    targets
                )

                B, num_queries, num_classes = pred_logits.shape
                tgt_classes = torch.full((B, num_queries), -1, dtype=torch.int64, device=self.device)
                for i, (pred_idx, tgt_idx) in enumerate(indices):
                    tgt_classes[i, pred_idx] = targets[i]['labels'][tgt_idx]
                loss_cls = self.class_loss_fn(pred_logits.view(-1, num_classes), tgt_classes.view(-1))

                tgt_boxes = torch.zeros_like(pred_boxes)
                for i, (pred_idx, tgt_idx) in enumerate(indices):
                    tgt_boxes[i, pred_idx] = targets[i]['boxes'][tgt_idx]
                loss_bbox = self.bbox_loss_fn(pred_boxes, tgt_boxes).mean()

                loss = loss_cls + self.config.get('bbox_loss_weight', 5.0) * loss_bbox
                total_loss += loss.item()
                if dist.is_initialized() and dist.get_rank() == 0:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.val_loader)
        if self.writer is not None:
            self.writer.add_scalar('val/loss_epoch', avg_loss, epoch)
        
        self.save_checkpoint(is_best=False)

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"🎉 New best model saved with val loss: {self.best_val_loss:.4f}")
            self.save_checkpoint(is_best=True)
            
        return avg_loss

    def save_checkpoint(self, is_best=False):
        ckpt_dir = Path(f"models/checkpoints/nuscenes_expert/{self.config['run_name']}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if dist.is_initialized() and dist.get_rank() != 0:
            return

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        last_payload = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        torch.save(last_payload, ckpt_dir / 'last_full.pth')

        if is_best:
            torch.save(model_to_save.state_dict(), ckpt_dir / 'best_model.pth')
            torch.save(last_payload, ckpt_dir / 'best_full.pth')
            
    def train(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"🚀 Starting training for NuScenes expert...")
            print(f"Device: {self.device}, Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Epoch {epoch+1}/{self.config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        if self.writer is not None:
            self.writer.close()
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("✅ Training completed!")

def main():
    parser = argparse.ArgumentParser(description='Train NuScenes Expert Model (DDP)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--run_name', type=str, default='run_001', help='A name for this training run for logging.')
    parser.add_argument('--cost_class', type=float, default=1.0, help='Classification cost for Hungarian matcher')
    parser.add_argument('--cost_bbox', type=float, default=5.0, help='BBox L1 cost for Hungarian matcher')
    parser.add_argument('--cost_giou', type=float, default=2.0, help='GIoU cost for Hungarian matcher')
    parser.add_argument('--bbox_loss_weight', type=float, default=5.0, help='Weight for bbox regression loss')
    parser.add_argument('--num_queries', type=int, default=100, help='Number of object queries for multi-object detection')
    parser.add_argument('--resume_from', type=str, default='', help='Path to checkpoint to resume from (best_model.pth, last_full.pth, best_full.pth, or state_dict)')
    parser.add_argument('--resume_mode', type=str, choices=['model','full'], default='model', help='Resume only model weights or full optimizer/scheduler state when a full checkpoint is provided')
    parser.add_argument('--local_rank', type=int, default=0, help='DDP: local GPU index')
    args = parser.parse_args()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device(args.device)

    config = vars(args)

    model = NuScenesExpert(num_queries=args.num_queries)
    base_train_loader = get_nuscenes_loader('train', batch_size=args.batch_size, num_workers=args.num_workers)
    base_val_loader = get_nuscenes_loader('val', batch_size=args.batch_size, num_workers=args.num_workers)
    train_dataset = base_train_loader.dataset
    val_dataset = base_val_loader.dataset

    if dist.is_initialized():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model.to(device)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            static_graph=False,
        )
    else:
        model.to(device)

    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=getattr(base_train_loader, 'collate_fn', None)
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=getattr(base_val_loader, 'collate_fn', None)
        )
    else:
        train_loader = base_train_loader
        val_loader = base_val_loader
    
    checkpoint = None
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume_from not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(state_dict, strict=True)

    if not dist.is_initialized() or dist.get_rank() == 0:
        config_dir = Path(f"models/configs/nuscenes_expert")
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / f"{args.run_name}_config.json", 'w') as f:
            json.dump(config, f, indent=2)

    trainer = NuScenesTrainer(model, train_loader, val_loader, device, config)
    if checkpoint is not None and args.resume_mode == 'full' and isinstance(checkpoint, dict):
        trainer.load_training_state(checkpoint)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
