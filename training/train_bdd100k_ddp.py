# training/train_bdd100k_ddp.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_iou, box_convert
from tqdm import tqdm
import json
from pathlib import Path
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models and dataloaders
from hungarian_matcher import HungarianMatcher
from models.experts import BDDDetectionExpert, BDDDrivableExpert, BDDSegmentationExpert
from dataloaders import get_bdd_detection_loader, get_bdd_drivable_loader, get_bdd_segmentation_loader

class BDDTrainer:
    def __init__(self, task, model, train_loader, val_loader, device, config):
        self.task = task
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
        
        # --- Loss Functions ---
        if self.task == 'detection':
            self.class_loss_fn = nn.CrossEntropyLoss(ignore_index=self.model.module.num_classes if hasattr(self.model, 'module') else self.model.num_classes)
            self.bbox_loss_fn = nn.SmoothL1Loss(reduction='sum')
            self.matcher = HungarianMatcher(
                cost_class=self.config.get('cost_class', 1.0),
                cost_bbox=self.config.get('cost_bbox', 5.0),
                cost_giou=self.config.get('cost_giou', 2.0)
            )
        else: 
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        
        # Only rank 0 should log to tensorboard
        if dist.is_initialized() and dist.get_rank() == 0:
            self.writer = SummaryWriter(f"models/runs/bdd100k_{self.task}_expert_{self.config['run_name']}")
        else:
            self.writer = None
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        # Set epoch for distributed samplers
        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        if hasattr(self.val_loader, 'sampler') and hasattr(self.val_loader.sampler, 'set_epoch'):
            self.val_loader.sampler.set_epoch(epoch)
        
        # Only show progress bar on rank 0
        if dist.is_initialized() and dist.get_rank() == 0:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")
        else:
            pbar = self.train_loader
        
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            
            if self.task == 'detection':
                loss = self._train_detection_batch(batch)
            else:
                loss = self._train_segmentation_batch(batch)
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step() # Step scheduler every batch for CosineAnnealingLR
            
            total_loss += loss.item()
            
            # Only update progress bar on rank 0
            if dist.is_initialized() and dist.get_rank() == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard (only on rank 0)
            if self.writer is not None:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss_batch', loss.item(), step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], step)
        
        avg_loss = total_loss / len(self.train_loader)
        if self.writer is not None:
            self.writer.add_scalar('train/loss_epoch', avg_loss, epoch)
        return avg_loss
    
    def _train_detection_batch(self, batch):
        images     = batch['image'].to(self.device)             # [B, 3, H, W]
        gt_boxes   = batch['bboxes'].to(self.device)            # [B, N_max, 4]
        gt_labels  = batch['labels'].to(self.device)            # [B, N_max]

        # Prepare a list of targets in DETR style
        targets = []
        for b in range(images.size(0)):
            # filter out padded entries
            mask = gt_labels[b] != -1
            targets.append({
                'boxes':  gt_boxes[b][mask],    # [Ni, 4]
                'labels': gt_labels[b][mask]     # [Ni]
            })

        # forward pass
        outputs = self.model(images)
        pred_logits = outputs['class_logits']  # [B, C, H, W]
        pred_boxes = outputs['bbox_deltas']   # [B, 4, H, W]
        
        # Reshape to [B, Q, C] format for Hungarian matcher
        B, C, H, W = pred_logits.shape
        Q = H * W
        pred_logits = pred_logits.permute(0, 2, 3, 1).reshape(B, Q, C)  # [B, Q, C]
        pred_boxes = pred_boxes.permute(0, 2, 3, 1).reshape(B, Q, 4)    # [B, Q, 4]

        # Convert ground truth from xyxy to cxcywh for Hungarian matcher
        targets_cxcywh = []
        for target in targets:
            if target['boxes'].numel() > 0:
                boxes_cxcywh = box_convert(target['boxes'], 'xyxy', 'cxcywh')
            else:
                boxes_cxcywh = target['boxes']
            targets_cxcywh.append({
                'boxes': boxes_cxcywh,
                'labels': target['labels']
            })

        # match predictions ⇄ targets
        indices = self.matcher({'pred_logits': pred_logits, 'pred_boxes': pred_boxes}, targets_cxcywh)

        # build loss
        # pred_logits and pred_boxes are already in [B, Q, C] and [B, Q, 4] format
        num_preds = pred_logits.shape[1]  # Q

        # flatten predictions
        pred_logits_flat = pred_logits.reshape(B * num_preds, C)
        pred_boxes_flat = pred_boxes.reshape(B * num_preds, 4)

        # create target tensors, initialized as "background" / zeros
        num_classes = self.model.module.num_classes if hasattr(self.model, 'module') else self.model.num_classes
        target_classes_flat = torch.full(
            (B * num_preds,), num_classes,
            dtype=torch.int64, device=self.device
        )
        target_boxes_flat = torch.zeros((B * num_preds, 4), dtype=torch.float32, device=self.device)

        # fill in only the matched indices
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            batch_offset = b * num_preds
            # class targets
            target_classes_flat[batch_offset + pred_idx] = targets_cxcywh[b]['labels'][tgt_idx]
            # bbox targets: already in cxcywh format
            target_boxes_flat[batch_offset + pred_idx] = targets_cxcywh[b]['boxes'][tgt_idx]

        # classification loss
        class_loss = self.class_loss_fn(
            pred_logits_flat, target_classes_flat
        )

        # bbox regression loss (only on matched preds)
        # we mask out the unmatched (still 0) entries so they don't contribute
        matched_mask = target_classes_flat != num_classes
        if matched_mask.any():
            bbox_loss = self.bbox_loss_fn(
                pred_boxes_flat[matched_mask],
                target_boxes_flat[matched_mask]
            )
        else:
            bbox_loss = torch.tensor(0.0, device=self.device)

        total_loss = class_loss + self.config.get('bbox_loss_weight', 2.0) * bbox_loss
        return total_loss

    def _train_segmentation_batch(self, batch):
        images = batch['image'].to(self.device)
        masks = batch['mask'].to(self.device)
        
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks)
        return loss

    @torch.no_grad()
    def _evaluate_detection_batch(self, batch):
        """
        Run the model on `batch`, compute:
         - loss (same as in training, but without backward)
         - simple metrics: avg IoU over matched boxes, recall@0.5 IoU
        Returns: loss_value (float), metrics (dict)
        """
        images    = batch['image'].to(self.device)       # [B,3,H,W]
        gt_boxes  = batch['bboxes'].to(self.device)      # [B, N_max, 4]
        gt_labels = batch['labels'].to(self.device)      # [B, N_max]

        # 1) build DETR-style targets
        targets = []
        for b in range(images.size(0)):
            mask = gt_labels[b] != -1
            targets.append({
                'boxes':  gt_boxes[b][mask],   # [Ni,4]
                'labels': gt_labels[b][mask]   # [Ni]
            })

        # 2) forward
        outputs = self.model(images)
        pred_logits = outputs['class_logits']  # [B, C, H, W]
        pred_boxes = outputs['bbox_deltas']   # [B, 4, H, W]
        
        # Reshape to [B, Q, C] format for Hungarian matcher
        B, C, H, W = pred_logits.shape
        Q = H * W
        pred_logits = pred_logits.permute(0, 2, 3, 1).reshape(B, Q, C)  # [B, Q, C]
        pred_boxes = pred_boxes.permute(0, 2, 3, 1).reshape(B, Q, 4)    # [B, Q, 4]

        # Convert ground truth from xyxy to cxcywh for Hungarian matcher
        targets_cxcywh = []
        for target in targets:
            if target['boxes'].numel() > 0:
                boxes_cxcywh = box_convert(target['boxes'], 'xyxy', 'cxcywh')
            else:
                boxes_cxcywh = target['boxes']
            targets_cxcywh.append({
                'boxes': boxes_cxcywh,
                'labels': target['labels']
            })

        # 3) match preds ⇄ targets
        indices = self.matcher(
            {'pred_logits': pred_logits, 'pred_boxes': pred_boxes},
            targets_cxcywh
        )

        # 4) compute loss exactly as in _train_detection_batch, but no backward
        # pred_logits and pred_boxes are already in [B, Q, C] and [B, Q, 4] format
        Q = pred_logits.shape[1]
        # flatten
        plog_flat = pred_logits.reshape(B*Q, C)
        pbbox_flat = pred_boxes.reshape(B*Q, 4)

        # init targets
        num_classes = self.model.module.num_classes if hasattr(self.model, 'module') else self.model.num_classes
        cls_tgt = torch.full((B*Q,), num_classes,
                             dtype=torch.int64, device=self.device)
        box_tgt = torch.zeros((B*Q, 4), dtype=torch.float32, device=self.device)

        for b, (p_idx, t_idx) in enumerate(indices):
            offset = b * Q
            cls_tgt[offset + p_idx] = targets_cxcywh[b]['labels'][t_idx]
            box_tgt[offset + p_idx] = targets_cxcywh[b]['boxes'][t_idx]

        # losses
        loss_cls = self.class_loss_fn(plog_flat, cls_tgt)
        mask    = cls_tgt != num_classes
        loss_box = ( self.bbox_loss_fn(
                        pbbox_flat[mask],
                        box_tgt[mask]
                    ) / mask.sum().clamp(min=1) )
        val_loss = loss_cls + self.config.get('bbox_loss_weight', 2.0)*loss_box

        # 5) simple IoU metric over matched preds
        iou_scores = []
        for b, (p_idx, t_idx) in enumerate(indices):
            if p_idx.numel()>0:
                # gather matched pred & gt boxes
                start = b*Q
                pr = pbbox_flat[start + p_idx]      # [M,4] in cxcywh?
                gt = box_tgt[start + p_idx]         # same format
                # convert to xyxy if needed
                pr_xy = box_convert(pr, 'cxcywh', 'xyxy')
                gt_xy = box_convert(gt, 'cxcywh', 'xyxy')
                # compute IoU matrix
                ious = box_iou(pr_xy, gt_xy).diagonal()  # [M]
                iou_scores.append(ious.mean().item())
        avg_iou = float(sum(iou_scores)/len(iou_scores)) if iou_scores else 0.0

        # 6) recall @0.5 as fraction of gt boxes having a matched pred IoU≥0.5
        recall_vals = []
        for b, (p_idx, t_idx) in enumerate(indices):
            if t_idx.numel()>0:
                # ious between all preds & this image's gt
                start = b*Q
                all_pr = pbbox_flat[start:start+Q]
                all_gt = targets_cxcywh[b]['boxes']
                pr_xy = box_convert(all_pr, 'cxcywh', 'xyxy')
                gt_xy = box_convert(all_gt, 'cxcywh', 'xyxy')
                mat = box_iou(pr_xy, gt_xy)  # [Q, Ni]
                # for each gt, does any pred IoU≥.5?
                match_gt = (mat.max(dim=0)[0] >= 0.5).float()
                recall_vals.append(match_gt.mean().item())
        recall50 = float(sum(recall_vals)/len(recall_vals)) if recall_vals else 0.0

        metrics = {
            'avg_iou': avg_iou,
            'recall_0.5': recall50
        }
        return val_loss.item(), metrics

    @torch.no_grad()
    def _evaluate_segmentation_batch(self, batch):
        """
        Evaluate one batch for segmentation or drivable tasks:
        - loss (CrossEntropy)
        - pixel accuracy (ignoring ignore_index)
        - mean IoU over classes (ignoring ignore_index)
        """
        images = batch['image'].to(self.device)     # [B, 3, H, W]
        masks  = batch['mask'].to(self.device)      # [B, H, W]
        
        # forward + loss
        outputs = self.model(images)                # [B, C, H, W]
        loss = self.loss_fn(outputs, masks)
        
        # predictions
        preds = outputs.argmax(dim=1)               # [B, H, W]
        ignore_mask = (masks == 255)
        
        # pixel accuracy
        valid = ~ignore_mask
        correct = (preds == masks) & valid
        pixel_acc = correct.sum().float() / valid.sum().clamp(min=1).float()
        
        # mean IoU
        num_classes = outputs.shape[1]
        ious = []
        for cls in range(num_classes):
            # skip if this class never appears in GT
            gt_cls = (masks == cls)
            if gt_cls.sum() == 0:
                continue
            pred_cls = (preds == cls)
            # intersection & union, excluding ignore
            inter = (pred_cls & gt_cls).sum().float()
            union = ((pred_cls | gt_cls) & ~ignore_mask).sum().float()
            ious.append((inter / union).item())
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        
        metrics = {
            'pixel_acc': pixel_acc.item(),
            'mean_iou': mean_iou
        }
        return loss.item(), metrics
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        # collect metrics per batch
        if self.task == 'detection':
            agg = {'avg_iou': [], 'recall_0.5': []}
        else:
            agg = {'pixel_acc': [], 'mean_iou': []}
        
        # Only show progress bar on rank 0
        if dist.is_initialized() and dist.get_rank() == 0:
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        else:
            pbar = self.val_loader
            
        for batch in pbar:
            if self.task == 'detection':
                loss, mets = self._evaluate_detection_batch(batch)
            else: 
                loss, mets = self._evaluate_segmentation_batch(batch)
            
            total_loss += loss
            for k,v in mets.items():
                agg[k].append(v)
            
            # Only update progress bar on rank 0
            if dist.is_initialized() and dist.get_rank() == 0:
                pbar.set_postfix({
                    'val_loss': f"{loss:.4f}",
                    **{k: f"{v:.3f}" for k,v in mets.items()}
                })
        
        avg_loss = total_loss / len(self.val_loader)
        # mean metrics across all batches
        final = {k: sum(vs)/len(vs) for k,vs in agg.items()}
        
        # Log to tensorboard (only on rank 0)
        if self.writer is not None:
            self.writer.add_scalar('val/loss_epoch', avg_loss, epoch)
            if self.task != 'detection':
                self.writer.add_scalar('val/pixel_acc', final['pixel_acc'], epoch)
                self.writer.add_scalar('val/mean_iou',  final['mean_iou'],  epoch)
            else:
                self.writer.add_scalar('val/avg_iou',   final['avg_iou'],   epoch)
                self.writer.add_scalar('val/recall_0.5', final['recall_0.5'], epoch)
        
        # checkpoint logic unchanged (only on rank 0)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"🎉 New best model saved with val loss: {self.best_val_loss:.4f}")
                self.save_checkpoint(epoch, is_best=True)
        
        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        # Only save on rank 0
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        
        # Synchronize before saving to prevent race conditions
        if dist.is_initialized():
            dist.barrier()
            
        ckpt_dir = Path(f"models/checkpoints/bdd100k_{self.task}_expert/{self.config['run_name']}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the underlying model (not the DDP wrapper)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if is_best:
            torch.save(checkpoint, ckpt_dir / "best.pth")
        
        # Synchronize after saving
        if dist.is_initialized():
            dist.barrier()
    
    def train(self):
        # Only print on rank 0
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"🚀 Starting training for BDD100K {self.task} expert...")
            print(f"Device: {self.device}, Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # Only print on rank 0
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"Epoch {epoch+1}/{self.config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if self.writer is not None:
            self.writer.close()
            
        # Only print on rank 0
        if dist.is_initialized() and dist.get_rank() == 0:
            print("✅ Training completed!")

def main():
    parser = argparse.ArgumentParser(description='Train BDD100K Expert Models')
    parser.add_argument('--task', type=str, required=True, choices=['detection', 'drivable', 'segmentation'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--run_name', type=str, default='run_001', help='A name for this training run for logging.')
    parser.add_argument('--cost_class', type=float, default=1.0, help='Classification cost for Hungarian matcher (detection only)')
    parser.add_argument('--cost_bbox', type=float, default=5.0, help='BBox L1 cost for Hungarian matcher (detection only)')
    parser.add_argument('--cost_giou', type=float, default=2.0, help='GIoU cost for Hungarian matcher (detection only)')
    parser.add_argument('--local_rank', type=int, default=0, help='DDP: local GPU index')
    args = parser.parse_args()
    
    # Initialize distributed training FIRST (before any model/optimizer construction)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        # Enable cuDNN benchmark for better performance
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device(args.device)
    
    config = vars(args)
    
    # --- Select Model and Dataloaders ---
    if args.task == 'detection':
        model = BDDDetectionExpert()
        # Get base loaders to preserve collate_fn
        base_train_loader = get_bdd_detection_loader('train', batch_size=args.batch_size, num_workers=args.num_workers)
        base_val_loader = get_bdd_detection_loader('val', batch_size=args.batch_size, num_workers=args.num_workers)
        train_dataset = base_train_loader.dataset
        val_dataset = base_val_loader.dataset
    elif args.task == 'drivable':
        model = BDDDrivableExpert()
        base_train_loader = get_bdd_drivable_loader('train', batch_size=args.batch_size, num_workers=args.num_workers)
        base_val_loader = get_bdd_drivable_loader('val', batch_size=args.batch_size, num_workers=args.num_workers)
        train_dataset = base_train_loader.dataset
        val_dataset = base_val_loader.dataset
    elif args.task == 'segmentation':
        model = BDDSegmentationExpert()
        base_train_loader = get_bdd_segmentation_loader('train', batch_size=args.batch_size, num_workers=args.num_workers)
        base_val_loader = get_bdd_segmentation_loader('val', batch_size=args.batch_size, num_workers=args.num_workers)
        train_dataset = base_train_loader.dataset
        val_dataset = base_val_loader.dataset
    
    # Wrap model in DDP if distributed training is enabled
    if dist.is_initialized():
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # Create distributed samplers and dataloaders
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=getattr(base_train_loader, 'collate_fn', None)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=getattr(base_val_loader, 'collate_fn', None)
        )
    else:
        # Use original dataloaders for non-distributed training
        train_loader = base_train_loader
        val_loader = base_val_loader
    
    # --- Save Config and Start Training ---
    if dist.is_initialized() and dist.get_rank() == 0:
        config_dir = Path(f"models/configs/bdd100k_{args.task}_expert")
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / f"{args.run_name}_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    trainer = BDDTrainer(args.task, model, train_loader, val_loader, device, config)
    trainer.train()
    
    # Clean up distributed training
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()

"""
torchrun --nproc_per_node=2 --standalone \
        training/train_bdd100k_ddp.py \
        --task detection \
        --epochs 50 \
        --batch_size 32 \
        --learning_rate 1e-4 \
        --run_name detection_ddp
"""