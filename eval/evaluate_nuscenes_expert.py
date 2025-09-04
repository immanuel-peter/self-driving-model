import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn as nn

from models.experts import NuScenesExpert
from dataloaders import get_nuscenes_loader
from training import HungarianMatcher

@torch.no_grad()
def evaluate(model: nn.Module, data_loader, device: torch.device, config: dict):
    model.eval()
    class_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    bbox_loss_fn  = nn.SmoothL1Loss(reduction='none')
    matcher = HungarianMatcher(
        cost_class=config.get('cost_class', 1.0),
        cost_bbox=config.get('cost_bbox', 5.0),
        cost_giou=config.get('cost_giou', 2.0),
    )

    total_loss = 0.0

    for batch in data_loader:
        batch_dict = {
            'image':      batch['image'].to(device),
            'lidar':      batch['lidar'].to(device),
            'intrinsics': batch['intrinsics'].to(device),
        }

        outputs = model(batch_dict)
        pred_logits = outputs['class_logits']  # [B, num_queries, 10]
        pred_boxes  = outputs['bbox_preds']    # [B, num_queries, 7]

        gt_boxes  = batch['boxes'].to(device)
        gt_labels = batch['labels'].to(device)

        targets = []
        B, M, _ = gt_boxes.shape
        for i in range(B):
            mask = gt_labels[i] != -1
            targets.append({
                'boxes':  gt_boxes[i][mask],
                'labels': gt_labels[i][mask],
            })

        indices = matcher({'pred_logits': pred_logits, 'pred_boxes': pred_boxes}, targets)

        B, num_queries, num_classes = pred_logits.shape
        tgt_classes = torch.full((B, num_queries), -1, dtype=torch.int64, device=device)
        for i, (pred_idx, tgt_idx) in enumerate(indices):
            tgt_classes[i, pred_idx] = targets[i]['labels'][tgt_idx]
        loss_cls = class_loss_fn(pred_logits.view(-1, num_classes), tgt_classes.view(-1))

        tgt_boxes = torch.zeros_like(pred_boxes)
        for i, (pred_idx, tgt_idx) in enumerate(indices):
            tgt_boxes[i, pred_idx] = targets[i]['boxes'][tgt_idx]
        loss_bbox = bbox_loss_fn(pred_boxes, tgt_boxes).mean()

        loss = loss_cls + config.get('bbox_loss_weight', 5.0) * loss_bbox
        total_loss += float(loss.item())

    n_batches = max(1, len(data_loader))
    return {'val_loss': total_loss / n_batches}


def main():
    parser = argparse.ArgumentParser(description='Evaluate NuScenes Expert')
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--split', choices=['val', 'train'], default='val')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_queries', type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')

    model = NuScenesExpert(num_queries=args.num_queries)
    ckpt_path = Path(f'models/checkpoints/nuscenes_expert/{args.run_name}/best_model.pth')
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    state_dict = torch.load(ckpt_path, map_location=str(device))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    loader = get_nuscenes_loader(args.split, batch_size=args.batch_size, num_workers=args.num_workers)

    metrics = evaluate(model, loader, device, config={})
    print({k: round(v, 4) for k, v in metrics.items()})

    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_path = results_dir / f"nuscenes_{args.run_name}_{args.split}_{ts}.json"
    payload = {
        "script": "evaluate_nuscenes_expert",
        "run_name": args.run_name,
        "split": args.split,
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_queries": args.num_queries,
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "metrics": metrics,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()


