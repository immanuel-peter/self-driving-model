import argparse
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torchvision.ops import box_convert
from PIL import Image, ImageDraw, ImageFont

# Local imports
from models.experts import BDDDetectionExpert
from dataloaders import get_bdd_detection_loader


@torch.no_grad()
def visualize_batch(model: nn.Module, images: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
                    out_dir: Path, device: torch.device, topk: int, score_thresh: float):
    model.eval()

    outputs = model(images.to(device))
    pred_logits = outputs["class_logits"]  # [B, C, H, W]
    pred_boxes = outputs["bbox_deltas"]    # [B, 4, H, W]

    B, C, Hf, Wf = pred_logits.shape
    Q = Hf * Wf

    # [B, Q, C] and [B, Q, 4]
    pred_logits = pred_logits.permute(0, 2, 3, 1).reshape(B, Q, C)
    pred_probs, pred_classes = pred_logits.softmax(-1).max(dim=-1)  # [B, Q]
    pred_boxes = pred_boxes.permute(0, 2, 3, 1).reshape(B, Q, 4)    # cx,cy,w,h (same convention as training/eval)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to load a default font; fallback to none
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for b in range(B):
        img = images[b].cpu().clamp(0, 1)  # [3,H,W]
        H, W = img.shape[1], img.shape[2]
        pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype('uint8'))
        draw = ImageDraw.Draw(pil)

        # Ground-truth: convert to xyxy for drawing
        gt_mask = (gt_labels[b] != -1)
        gt_xyxy = box_convert(gt_boxes[b][gt_mask].cpu(), 'xyxy', 'xyxy')  # identity, stored as xyxy in dataset
        for xy in gt_xyxy:
            x1, y1, x2, y2 = [float(v) for v in xy]
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

        # Predictions: take top-K by confidence and threshold
        scores = pred_probs[b].cpu()
        boxes = pred_boxes[b].cpu()
        classes = pred_classes[b].cpu()

        keep = scores >= score_thresh
        scores = scores[keep]
        boxes = boxes[keep]
        classes = classes[keep]

        if scores.numel() > 0:
            topk_idx = scores.topk(min(topk, scores.numel())).indices
            scores = scores[topk_idx]
            boxes = boxes[topk_idx]
            classes = classes[topk_idx]

            # Convert to xyxy for drawing; boxes are expected in cxcywh as in training/eval
            pred_xyxy = box_convert(boxes, 'cxcywh', 'xyxy')

            # Draw predictions in red
            for i, xy in enumerate(pred_xyxy):
                x1, y1, x2, y2 = [float(v) for v in xy]
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                label = f"c{int(classes[i])}:{scores[i]:.2f}"
                if font is not None:
                    draw.text((x1 + 2, y1 + 2), label, fill=(255, 0, 0), font=font)
                else:
                    draw.text((x1 + 2, y1 + 2), label, fill=(255, 0, 0))

        pil.save(out_dir / f"sample_{b:03d}.jpg")


def main():
    parser = argparse.ArgumentParser(description="Visualize BDD100K Detection predictions vs GT")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=16, help="Max number of images to visualize")
    parser.add_argument("--topk", type=int, default=100, help="Top-K predictions per image to draw")
    parser.add_argument("--score_thresh", type=float, default=0.30, help="Min confidence to draw a prediction")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Model and checkpoint
    model = BDDDetectionExpert()
    ckpt_path = Path(f"models/checkpoints/bdd100k_detection_expert/{args.run_name}/best.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=str(device))
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Data
    loader = get_bdd_detection_loader(args.split, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # Output directory
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = Path("eval/vis") / f"bdd100k_detection_{args.run_name}_{args.split}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over batches until limit is reached
    num_done = 0
    for batch in loader:
        images = batch["image"]
        gt_boxes = batch["bboxes"]
        gt_labels = batch["labels"]

        to_do = min(images.size(0), args.limit - num_done)
        if to_do <= 0:
            break

        visualize_batch(model, images[:to_do], gt_boxes[:to_do], gt_labels[:to_do],
                        out_dir, device, args.topk, args.score_thresh)
        num_done += to_do

        if num_done >= args.limit:
            break

    print(f"Saved {num_done} visualizations to {out_dir}")


if __name__ == "__main__":
    main()


