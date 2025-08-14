import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.ops import box_convert, box_iou

# Local imports
from models.experts import (
    BDDDetectionExpert,
    BDDDrivableExpert,
    BDDSegmentationExpert,
)
from dataloaders import (
    get_bdd_detection_loader,
    get_bdd_drivable_loader,
    get_bdd_segmentation_loader,
)
from training import HungarianMatcher


@torch.no_grad()
def evaluate_detection(model: nn.Module, data_loader, device: torch.device, config: dict):
    model.eval()
    matcher = HungarianMatcher(
        cost_class=config.get("cost_class", 1.0),
        cost_bbox=config.get("cost_bbox", 5.0),
        cost_giou=config.get("cost_giou", 2.0),
    )

    class_loss_fn = nn.CrossEntropyLoss(ignore_index=model.num_classes)
    bbox_loss_fn = nn.SmoothL1Loss(reduction="sum")

    total_loss = 0.0
    agg_iou = []
    agg_recall50 = []

    for batch in data_loader:
        images = batch["image"].to(device)
        gt_boxes = batch["bboxes"].to(device)
        gt_labels = batch["labels"].to(device)

        # Build DETR-style targets
        targets = []
        for b in range(images.size(0)):
            mask = gt_labels[b] != -1
            targets.append({
                "boxes": gt_boxes[b][mask],
                "labels": gt_labels[b][mask],
            })

        # Forward
        outputs = model(images)
        pred_logits = outputs["class_logits"]  # [B, C, H, W]
        pred_boxes = outputs["bbox_deltas"]    # [B, 4, H, W]

        # Reshape to [B, Q, C] / [B, Q, 4]
        B, C, H, W = pred_logits.shape
        Q = H * W
        pred_logits = pred_logits.permute(0, 2, 3, 1).reshape(B, Q, C)
        pred_boxes = pred_boxes.permute(0, 2, 3, 1).reshape(B, Q, 4)

        # Convert GT boxes to cxcywh for matching
        targets_cxcywh = []
        for t in targets:
            if t["boxes"].numel() > 0:
                boxes_cxcywh = box_convert(t["boxes"], "xyxy", "cxcywh")
            else:
                boxes_cxcywh = t["boxes"]
            targets_cxcywh.append({
                "boxes": boxes_cxcywh,
                "labels": t["labels"],
            })

        # Match
        indices = matcher({"pred_logits": pred_logits, "pred_boxes": pred_boxes}, targets_cxcywh)

        # Compute losses (same formulation used during training)
        pred_logits_flat = pred_logits.reshape(B * Q, C)
        pred_boxes_flat = pred_boxes.reshape(B * Q, 4)
        num_classes = model.num_classes

        target_classes_flat = torch.full((B * Q,), num_classes, dtype=torch.int64, device=device)
        target_boxes_flat = torch.zeros((B * Q, 4), dtype=torch.float32, device=device)
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            offset = b * Q
            if pred_idx.numel() > 0:
                target_classes_flat[offset + pred_idx] = targets_cxcywh[b]["labels"][tgt_idx]
                target_boxes_flat[offset + pred_idx] = targets_cxcywh[b]["boxes"][tgt_idx]

        loss_cls = class_loss_fn(pred_logits_flat, target_classes_flat)
        matched_mask = target_classes_flat != num_classes
        if matched_mask.any():
            loss_bbox = bbox_loss_fn(
                pred_boxes_flat[matched_mask],
                target_boxes_flat[matched_mask]
            )
        else:
            loss_bbox = torch.tensor(0.0, device=device)
        loss = loss_cls + config.get("bbox_loss_weight", 2.0) * loss_bbox
        total_loss += float(loss.item())

        # Metrics: avg IoU on matched + recall@0.5
        # avg IoU
        iou_scores = []
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() > 0:
                start = b * Q
                pr = pred_boxes_flat[start + pred_idx]
                gt = target_boxes_flat[start + pred_idx]
                pr_xy = box_convert(pr, "cxcywh", "xyxy")
                gt_xy = box_convert(gt, "cxcywh", "xyxy")
                ious = box_iou(pr_xy, gt_xy).diagonal()
                iou_scores.append(float(ious.mean().item()))
        avg_iou = (sum(iou_scores) / len(iou_scores)) if iou_scores else 0.0
        agg_iou.append(avg_iou)

        # recall@0.5
        recall_vals = []
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if tgt_idx.numel() > 0:
                start = b * Q
                all_pr = pred_boxes_flat[start : start + Q]
                all_gt = targets_cxcywh[b]["boxes"]
                pr_xy = box_convert(all_pr, "cxcywh", "xyxy")
                gt_xy = box_convert(all_gt, "cxcywh", "xyxy")
                mat = box_iou(pr_xy, gt_xy)
                match_gt = (mat.max(dim=0)[0] >= 0.5).float()
                recall_vals.append(float(match_gt.mean().item()))
        recall50 = (sum(recall_vals) / len(recall_vals)) if recall_vals else 0.0
        agg_recall50.append(recall50)

    n_batches = max(1, len(data_loader))
    return {
        "val_loss": total_loss / n_batches,
        "avg_iou": sum(agg_iou) / len(agg_iou) if agg_iou else 0.0,
        "recall_0.5": sum(agg_recall50) / len(agg_recall50) if agg_recall50 else 0.0,
    }


@torch.no_grad()
def evaluate_seg_like(model: nn.Module, data_loader, device: torch.device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    total_loss = 0.0
    agg_pixel_acc = []
    agg_mean_iou = []

    for batch in data_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)  # [B, C, H, W]
        loss = loss_fn(outputs, masks)
        total_loss += float(loss.item())

        preds = outputs.argmax(dim=1)  # [B, H, W]
        ignore_mask = (masks == 255)
        valid = ~ignore_mask
        correct = (preds == masks) & valid
        pixel_acc = (correct.sum().float() / valid.sum().clamp(min=1).float()).item()
        agg_pixel_acc.append(pixel_acc)

        # mean IoU
        num_classes = outputs.shape[1]
        ious = []
        for cls in range(num_classes):
            gt_cls = (masks == cls)
            if gt_cls.sum() == 0:
                continue
            pred_cls = (preds == cls)
            inter = (pred_cls & gt_cls).sum().float()
            union = ((pred_cls | gt_cls) & ~ignore_mask).sum().float()
            if union.item() > 0:
                ious.append((inter / union).item())
        mean_iou = (sum(ious) / len(ious)) if ious else 0.0
        agg_mean_iou.append(mean_iou)

    n_batches = max(1, len(data_loader))
    return {
        "val_loss": total_loss / n_batches,
        "pixel_acc": sum(agg_pixel_acc) / len(agg_pixel_acc) if agg_pixel_acc else 0.0,
        "mean_iou": sum(agg_mean_iou) / len(agg_mean_iou) if agg_mean_iou else 0.0,
    }


def load_bdd_checkpoint(model: nn.Module, task: str, run_name: str, map_location: str = "cpu"):
    ckpt_path = Path(f"models/checkpoints/bdd100k_{task}_expert/{run_name}/best.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate BDD100K Experts")
    parser.add_argument("--task", choices=["detection", "drivable", "segmentation"], required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.task == "detection":
        model = BDDDetectionExpert()
        base_loader = get_bdd_detection_loader(args.split, batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.task == "drivable":
        model = BDDDrivableExpert()
        base_loader = get_bdd_drivable_loader(args.split, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        model = BDDSegmentationExpert()
        base_loader = get_bdd_segmentation_loader(args.split, batch_size=args.batch_size, num_workers=args.num_workers)

    model.to(device)
    ckpt = load_bdd_checkpoint(model, args.task, args.run_name, map_location=str(device))

    # Pull training-time config if present for metric weights
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    if args.task == "detection":
        metrics = evaluate_detection(model, base_loader, device, config)
    else:
        metrics = evaluate_seg_like(model, base_loader, device)

    # Pretty print to console
    print({k: round(v, 4) for k, v in metrics.items()})

    # Persist results to eval/results/*.json
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_path = results_dir / f"bdd100k_{args.task}_{args.run_name}_{args.split}_{ts}.json"
    payload = {
        "script": "evaluate_bdd100k_expert",
        "task": args.task,
        "run_name": args.run_name,
        "split": args.split,
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "metrics": metrics,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()


