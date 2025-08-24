import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torchvision.ops import box_convert
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hungarian_matcher import HungarianMatcher
from models.experts import BDDDetectionExpert, BDDSegmentationExpert, BDDDrivableExpert
from dataloaders.carla_detection_loader import CarlaDetectionDataset, detection_collate_fn
from dataloaders.carla_segmentation_loader import CarlaSegmentationDataset
from dataloaders.carla_drivable_loader import CarlaDrivableDataset

DEFAULT_CARLA_PREPROCESSED_ROOT = "datasets/carla/preprocessed"

def setup_ddp(rank: int, world_size: int, backend: str = 'nccl'):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    def __init__(self, task: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, config: dict, rank: int):
        self.task = task
        self.model = model.to(device)
        base_model = self.model.module if isinstance(self.model, DDP) else self.model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.rank = rank

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, config['epochs']) * max(1, len(self.train_loader)))

        if self.task == 'detection':
            self.num_classes = getattr(base_model, 'num_classes', None)
            if self.num_classes is None:
                raise AttributeError('Detection model must expose num_classes')
            self.class_loss_fn = nn.CrossEntropyLoss(ignore_index=self.num_classes)
            self.bbox_loss_fn = nn.SmoothL1Loss(reduction='mean')
            self.matcher = HungarianMatcher(
                cost_class=self.config.get('cost_class', 1.0),
                cost_bbox=self.config.get('cost_bbox', 5.0),
                cost_giou=self.config.get('cost_giou', 2.0)
            )
        else:
            # For segmentation/drivable tasks, cache class count and loss
            self.seg_num_classes = getattr(base_model, 'num_classes', None)
            if self.seg_num_classes is None:
                raise AttributeError('Segmentation model must expose num_classes')
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    def _train_detection_batch(self, batch):
        images = batch['image'].to(self.device)
        gt_boxes_xyxy = batch['bboxes'].to(self.device)
        gt_labels = batch['labels'].to(self.device)

        targets = []
        for b in range(images.size(0)):
            mask = gt_labels[b] != -1
            targets.append({'boxes': gt_boxes_xyxy[b][mask], 'labels': gt_labels[b][mask]})

        outputs = self.model(images)
        pred_logits = outputs['class_logits']  # [B,C,H,W]
        pred_boxes = outputs['bbox_deltas']    # [B,4,H,W]

        B, C, H, W = pred_logits.shape
        Q = H * W
        pred_logits = pred_logits.permute(0, 2, 3, 1).reshape(B, Q, C)
        pred_boxes = pred_boxes.permute(0, 2, 3, 1).reshape(B, Q, 4)

        # GT to cxcywh
        targets_cxcywh = []
        for t in targets:
            boxes = t['boxes']
            if boxes.numel() > 0:
                boxes = box_convert(boxes, 'xyxy', 'cxcywh')
            targets_cxcywh.append({'boxes': boxes, 'labels': t['labels']})

        indices = self.matcher({'pred_logits': pred_logits, 'pred_boxes': pred_boxes}, targets_cxcywh)

        num_preds = pred_logits.shape[1]
        pred_logits_flat = pred_logits.reshape(B * num_preds, C)
        pred_boxes_flat = pred_boxes.reshape(B * num_preds, 4)
        target_classes_flat = torch.full((B * num_preds,), self.num_classes, dtype=torch.int64, device=self.device)
        target_boxes_flat = torch.zeros((B * num_preds, 4), dtype=torch.float32, device=self.device)

        for b, (pi, ti) in enumerate(indices):
            off = b * num_preds
            if pi.numel() > 0:
                target_classes_flat[off + pi] = targets_cxcywh[b]['labels'][ti]
                target_boxes_flat[off + pi] = targets_cxcywh[b]['boxes'][ti]

        # Robust classification loss: skip positions with background (ignore)
        valid_cls = target_classes_flat != self.num_classes
        if valid_cls.any():
            cls_loss = F.cross_entropy(
                pred_logits_flat[valid_cls],
                target_classes_flat[valid_cls],
                reduction='mean'
            )
        else:
            cls_loss = torch.tensor(0.0, device=self.device)

        mask = valid_cls
        if mask.any():
            box_loss = self.bbox_loss_fn(pred_boxes_flat[mask], target_boxes_flat[mask])
        else:
            box_loss = torch.tensor(0.0, device=self.device)
        return cls_loss + self.config.get('bbox_loss_weight', 1.0) * box_loss

    def _train_segmentation_batch(self, batch):
        images = batch['image'].to(self.device)
        masks = batch['mask'].to(self.device)
        # Sanitize labels: set anything outside [0, num_classes-1] to ignore (255)
        if masks.dim() == 4:
            # In case any loader returns [B,H,W,C], take first channel
            masks = masks[..., 0]
        invalid = (masks < 0) | (masks >= self.seg_num_classes)
        if invalid.any():
            masks = masks.clone()
            masks[invalid] = 255
        logits = self.model(images)
        return self.loss_fn(logits, masks)

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        total = 0.0
        start = time.time()
        # Only rank 0 shows a tqdm with epoch info
        desc = f"train | epoch {epoch_idx+1}/{self.config['epochs']}"
        iterable = self.train_loader if self.rank != 0 else tqdm(self.train_loader, desc=desc)
        for batch in iterable:
            self.optimizer.zero_grad()
            if self.task == 'detection':
                loss = self._train_detection_batch(batch)
            else:
                try:
                    loss = self._train_segmentation_batch(batch)
                except Exception as e:
                    if self.rank == 0:
                        print(f"segmentation batch error: {repr(e)}")
                    raise
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            total += float(loss.item())
        dur = time.time() - start
        avg = total / max(1, len(self.train_loader))
        if self.rank == 0:
            print(f"epoch {epoch_idx+1} train_avg_loss={avg:.4f} time_s={dur:.1f}")
        return avg

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total = 0.0
        start = time.time()
        for batch in self.val_loader:
            if self.task == 'detection':
                total += float(self._train_detection_batch(batch).item())
            else:
                total += float(self._train_segmentation_batch(batch).item())
        dur = time.time() - start
        avg = total / max(1, len(self.val_loader))
        if self.rank == 0:
            print(f"val_avg_loss={avg:.4f} time_s={dur:.1f}")
        return avg


def ddp_worker(rank: int, world_size: int, args):
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')

    # Datasets and samplers
    if args.task == 'detection':
        train_set = CarlaDetectionDataset('train', root_dir=args.data_root)
        val_set = CarlaDetectionDataset('val', root_dir=args.data_root)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True), num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=detection_collate_fn)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False), num_workers=args.num_workers, pin_memory=True, drop_last=False, collate_fn=detection_collate_fn)
        model = BDDDetectionExpert(pretrained_backbone=True).to(device)
    elif args.task == 'segmentation':
        train_set = CarlaSegmentationDataset('train', root_dir=args.data_root)
        val_set = CarlaSegmentationDataset('val', root_dir=args.data_root)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True), num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False), num_workers=args.num_workers, pin_memory=True, drop_last=False)
        model = BDDSegmentationExpert(pretrained_backbone=True).to(device)
    elif args.task == 'drivable':
        train_set = CarlaDrivableDataset('train', root_dir=args.data_root)
        val_set = CarlaDrivableDataset('val', root_dir=args.data_root)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True), num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False), num_workers=args.num_workers, pin_memory=True, drop_last=False)
        model = BDDDrivableExpert(pretrained_backbone=True).to(device)

    # Wrap DDP
    model = DDP(model, device_ids=[rank] if device.type == 'cuda' else None, output_device=rank if device.type == 'cuda' else None, find_unused_parameters=False)

    trainer = Trainer(args.task, model, train_loader, val_loader, device, vars(args), rank)

    best = float('inf')
    run_start = time.time()
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        trainer.train_epoch(epoch)
        val = trainer.validate()
        # Reduce best val across ranks (take min)
        t = torch.tensor([val], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        best = min(best, float(t.item()))

    # Save only on rank 0
    if rank == 0:
        total_time = time.time() - run_start
        ckpt_dir = Path(f"models/checkpoints/carla_{args.task}_expert_ddp/{args.run_name}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'best_val_loss': best,
            'config': vars(args),
        }, ckpt_dir / 'best.pth')

        cfg_dir = Path(f"models/configs/carla_{args.task}_expert_ddp")
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / f"{args.run_name}_config.json", 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f"completed {args.epochs} epochs | best_val_loss={best:.4f} | total_time_s={total_time:.1f}")

    cleanup_ddp()


def parse_args():
    parser = argparse.ArgumentParser(description='DDP: Fine-tune BDD experts on CARLA')
    parser.add_argument('--task', type=str, required=True, choices=['detection', 'segmentation', 'drivable'])
    parser.add_argument('--data_root', type=str, default=DEFAULT_CARLA_PREPROCESSED_ROOT)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--bbox_loss_weight', type=float, default=1.0)
    parser.add_argument('--cost_class', type=float, default=1.0)
    parser.add_argument('--cost_bbox', type=float, default=5.0)
    parser.add_argument('--cost_giou', type=float, default=2.0)
    parser.add_argument('--run_name', type=str, default='carla_ft_ddp')
    # torchrun env inferred: LOCAL_RANK / RANK / WORLD_SIZE, we use torchrun to spawn
    return parser.parse_args()


def main():
    args = parse_args()
    # torchrun will set WORLD_SIZE and LOCAL_RANK; using spawn fallback if missing
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size > 1:
        ddp_worker(local_rank, world_size, args)
    else:
        # Single process (still run the same worker path)
        ddp_worker(0, 1, args)


if __name__ == '__main__':
    main()


