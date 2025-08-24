import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torchvision.ops import box_convert
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hungarian_matcher import HungarianMatcher
from models.experts.nuscenes_expert import NuScenesExpert
from dataloaders.carla_detection_loader import CarlaDetectionDataset, detection_collate_fn

DEFAULT_CARLA_PREPROCESSED_ROOT = "datasets/carla/preprocessed"

class ImageOnlyWrapper(nn.Module):
    def __init__(self, base: NuScenesExpert, num_queries: int = 196, num_classes: int = 10):
        super().__init__()
        self.image_backbone = base.image_backbone
        self.image_projection = base.image_projection
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.query_embed = nn.Embedding(num_queries, 256)
        self.mlp = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
        )
        self.class_head = nn.Linear(128, num_classes)
        self.box_head = nn.Linear(128, 4)

    def forward(self, images: torch.Tensor):
        feat = self.image_backbone(images)
        feat = feat.view(feat.size(0), -1)
        feat = self.image_projection(feat)
        B = feat.size(0)
        fused = feat.unsqueeze(1).expand(B, self.num_queries, -1) + self.query_embed.weight.unsqueeze(0)
        x = self.mlp(fused)
        logits = self.class_head(x)
        boxes = self.box_head(x)
        return {"pred_logits": logits, "pred_boxes": boxes}


class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, config: dict, rank: int):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.rank = rank

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, config['epochs']) * max(1, len(self.train_loader)))
        self.matcher = HungarianMatcher(cost_class=config.get('cost_class', 1.0), cost_bbox=config.get('cost_bbox', 5.0), cost_giou=config.get('cost_giou', 2.0))
        self.class_loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.get('num_classes', 10))
        self.bbox_loss_fn = nn.SmoothL1Loss(reduction='mean')

    def _step(self, batch):
        images = batch['image'].to(self.device)
        gt_xyxy = batch['bboxes'].to(self.device)
        gt_labels = batch['labels'].to(self.device)

        out = self.model(images)
        pred_logits = out['pred_logits']
        pred_boxes = out['pred_boxes']

        B, Q, C = pred_logits.shape
        targets = []
        for b in range(B):
            mask = gt_labels[b] != -1
            boxes = gt_xyxy[b][mask]
            if boxes.numel() > 0:
                boxes = box_convert(boxes, 'xyxy', 'cxcywh')
            targets.append({'boxes': boxes, 'labels': gt_labels[b][mask]})

        indices = self.matcher({'pred_logits': pred_logits, 'pred_boxes': pred_boxes}, targets)

        pred_logits_flat = pred_logits.reshape(B * Q, C)
        pred_boxes_flat = pred_boxes.reshape(B * Q, 4)
        bg_id = self.config.get('num_classes', 10)
        target_classes_flat = torch.full((B * Q,), bg_id, dtype=torch.int64, device=self.device)
        target_boxes_flat = torch.zeros((B * Q, 4), dtype=torch.float32, device=self.device)

        for b, (pi, ti) in enumerate(indices):
            off = b * Q
            if pi.numel() > 0:
                target_classes_flat[off + pi] = targets[b]['labels'][ti]
                target_boxes_flat[off + pi] = targets[b]['boxes'][ti]

        # Robust classification loss: compute over valid (non-background-ignored) targets
        valid_cls = target_classes_flat != bg_id
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

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        total = 0.0
        start = time.time()
        desc = f"train | epoch {epoch_idx+1}/{self.config['epochs']}"
        iterable = self.train_loader if self.rank != 0 else tqdm(self.train_loader, desc=desc)
        for batch in iterable:
            self.optimizer.zero_grad()
            loss = self._step(batch)
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
            total += float(self._step(batch).item())
        dur = time.time() - start
        avg = total / max(1, len(self.val_loader))
        if self.rank == 0:
            print(f"val_avg_loss={avg:.4f} time_s={dur:.1f}")
        return avg


def setup_ddp(rank: int, world_size: int, backend: str = 'nccl'):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29501')
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def ddp_worker(rank: int, world_size: int, args):
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')

    train_set = CarlaDetectionDataset('train', root_dir=args.data_root)
    val_set = CarlaDetectionDataset('val', root_dir=args.data_root)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True), num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False), num_workers=args.num_workers, pin_memory=True, drop_last=False, collate_fn=detection_collate_fn)

    base = NuScenesExpert(num_queries=100)
    model = ImageOnlyWrapper(base, num_queries=args.num_queries, num_classes=10).to(device)
    model = DDP(model, device_ids=[rank] if device.type == 'cuda' else None, output_device=rank if device.type == 'cuda' else None, find_unused_parameters=False)

    trainer = Trainer(model, train_loader, val_loader, device, vars(args), rank)
    best = float('inf')
    run_start = time.time()
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        trainer.train_epoch(epoch)
        val = trainer.validate()
        # Reduce best val across ranks
        t = torch.tensor([val], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        best = min(best, float(t.item()))

    if rank == 0:
        total_time = time.time() - run_start
        ckpt_dir = Path(f"models/checkpoints/carla_nuscenes_2d_ddp/{args.run_name}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'best_val_loss': best,
            'config': vars(args),
        }, ckpt_dir / 'best.pth')

        cfg_dir = Path("models/configs/carla_nuscenes_2d_ddp")
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / f"{args.run_name}_config.json", 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f"completed {args.epochs} epochs | best_val_loss={best:.4f} | total_time_s={total_time:.1f}")

    cleanup_ddp()


def parse_args():
    parser = argparse.ArgumentParser(description='DDP: NuScenes image-only 2D fine-tune on CARLA')
    parser.add_argument('--data_root', type=str, default=DEFAULT_CARLA_PREPROCESSED_ROOT)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--bbox_loss_weight', type=float, default=1.0)
    parser.add_argument('--cost_class', type=float, default=1.0)
    parser.add_argument('--cost_bbox', type=float, default=5.0)
    parser.add_argument('--cost_giou', type=float, default=2.0)
    parser.add_argument('--num_queries', type=int, default=196)
    parser.add_argument('--run_name', type=str, default='nuscenes_img_only_carla_ft_ddp')
    return parser.parse_args()


def main():
    args = parse_args()
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size > 1:
        ddp_worker(local_rank, world_size, args)
    else:
        ddp_worker(0, 1, args)


if __name__ == '__main__':
    main()


