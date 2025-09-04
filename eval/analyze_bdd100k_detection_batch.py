import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torchvision.ops import box_convert, box_iou

from models.experts import BDDDetectionExpert
from dataloaders import get_bdd_detection_loader
from training import HungarianMatcher


@torch.no_grad()
def analyze_detection(model: nn.Module, data_loader, device: torch.device, config: dict, limit: int):
    model.eval()
    matcher = HungarianMatcher(
        cost_class=config.get("cost_class", 1.0),
        cost_bbox=config.get("cost_bbox", 5.0),
        cost_giou=config.get("cost_giou", 2.0),
    )

    per_image = []
    num_done = 0

    for batch in data_loader:
        images = batch["image"].to(device)
        gt_boxes = batch["bboxes"].to(device)    # [B, N, 4] in xyxy
        gt_labels = batch["labels"].to(device)   # [B, N]

        targets = []
        for b in range(images.size(0)):
            mask = gt_labels[b] != -1
            boxes_xyxy = gt_boxes[b][mask]
            boxes_cxcywh = box_convert(boxes_xyxy, "xyxy", "cxcywh") if boxes_xyxy.numel() > 0 else boxes_xyxy
            targets.append({
                "boxes": boxes_cxcywh,
                "labels": gt_labels[b][mask],
            })

        outputs = model(images)
        pred_logits = outputs["class_logits"]  # [B, C, H, W]
        pred_boxes = outputs["bbox_deltas"]    # [B, 4, H, W]

        B, C, H, W = pred_logits.shape
        Q = H * W
        pred_logits = pred_logits.permute(0, 2, 3, 1).reshape(B, Q, C)
        pred_boxes = pred_boxes.permute(0, 2, 3, 1).reshape(B, Q, 4)

        indices = matcher({"pred_logits": pred_logits, "pred_boxes": pred_boxes}, targets)

        for b in range(B):
            if num_done >= limit:
                break

            pred_idx, tgt_idx = indices[b]
            # Matched IoU
            avg_iou = 0.0
            if pred_idx.numel() > 0:
                pr_cxcywh = pred_boxes[b][pred_idx]
                gt_cxcywh = targets[b]["boxes"][tgt_idx]
                pr_xyxy = box_convert(pr_cxcywh, "cxcywh", "xyxy")
                gt_xyxy = box_convert(gt_cxcywh, "cxcywh", "xyxy")
                ious = box_iou(pr_xyxy, gt_xyxy).diagonal()
                avg_iou = float(ious.mean().item()) if ious.numel() > 0 else 0.0

            # Recall@0.5
            recall50 = 0.0
            if targets[b]["boxes"].numel() > 0:
                all_pr_xyxy = box_convert(pred_boxes[b], "cxcywh", "xyxy")
                all_gt_xyxy = box_convert(targets[b]["boxes"], "cxcywh", "xyxy")
                mat = box_iou(all_pr_xyxy, all_gt_xyxy)
                match_gt = (mat.max(dim=0)[0] >= 0.5).float()
                recall50 = float(match_gt.mean().item())

            per_image.append({
                "index": num_done,
                "num_gt": int((gt_labels[b] != -1).sum().item()),
                "num_matched": int(pred_idx.numel()),
                "avg_iou_matched": avg_iou,
                "recall_0.5": recall50,
            })
            num_done += 1

        if num_done >= limit:
            break

    return per_image


def main():
    parser = argparse.ArgumentParser(description="Analyze first N BDD100K detection samples")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = BDDDetectionExpert()
    ckpt_path = Path(f"models/checkpoints/bdd100k_detection_expert/{args.run_name}/best.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=str(device))
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    loader = get_bdd_detection_loader(args.split, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    per_image = analyze_detection(model, loader, device, config, args.limit)

    print("idx\tnGT\tnMatch\tIoU(avg)\tRecall@0.5")
    for r in per_image:
        print(f"{r['index']}\t{r['num_gt']}\t{r['num_matched']}\t{r['avg_iou_matched']:.3f}\t{r['recall_0.5']:.3f}")

    # Save to results
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_path = results_dir / f"bdd100k_detection_{args.run_name}_{args.split}_per_image_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"per_image": per_image}, f, indent=2)
    print(f"Saved per-image metrics to {out_path}")


if __name__ == "__main__":
    main()


