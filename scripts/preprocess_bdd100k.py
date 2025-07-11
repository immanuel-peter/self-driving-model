import json
import argparse
from tqdm import tqdm
from pathlib import Path
import torch

# Map BDD100K detection categories to integer IDs
CATEGORY_TO_ID = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic light": 8,
    "traffic sign": 9,
}

def parse_label(label):
    if "box2d" not in label or label["category"] not in CATEGORY_TO_ID:
        return None
    box = label["box2d"]
    bbox = [box["x1"], box["y1"], box["x2"], box["y2"]]
    label_id = CATEGORY_TO_ID[label["category"]]
    return bbox, label_id

def process_detection(json_path, image_root, save_dir):
    with open(json_path, "r") as f:
        annotations = json.load(f)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(annotations, desc=f"Processing {json_path}"):
        image_name = item["name"]
        image_path = str(Path(image_root) / image_name)

        bboxes = []
        labels = []

        for label in item.get("labels", []):
            parsed = parse_label(label)
            if parsed:
                bbox, label_id = parsed
                bboxes.append(bbox)
                labels.append(label_id)

        if not bboxes:
            continue

        sample = {
            "image_path": image_path,
            "bboxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "meta": {
                "scene": item.get("attributes", {}).get("scene", ""),
                "timeofday": item.get("attributes", {}).get("timeofday", ""),
                "weather": item.get("attributes", {}).get("weather", "")
            }
        }

        save_path = save_dir / (Path(image_name).stem + ".pt")
        torch.save(sample, save_path)

def process_segmentation(image_dir, mask_dir, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(Path(mask_dir).glob("*.png"))

    for mask_path in tqdm(mask_files, desc=f"Processing {mask_dir}"):
        stem = mask_path.stem
        image_path = str(Path(image_dir) / (stem + ".jpg"))

        sample = {
            "image_path": image_path,
            "mask_path": str(mask_path)
        }

        save_path = save_dir / (stem + ".pt")
        torch.save(sample, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["detection", "drivable", "segmentation"])
    parser.add_argument("--raw_dir", type=str, default="datasets/bdd100k/raw")
    parser.add_argument("--out_dir", type=str, default="datasets/bdd100k/preprocessed")
    args = parser.parse_args()

    image_root = Path(args.raw_dir) / "images" / "100k"
    out_root = Path(args.out_dir) / args.task

    if args.task == "detection":
        label_root = Path(args.raw_dir) / "labels" / "detection2020"
        process_detection(label_root / "det_train.json", image_root / "train", out_root / "train")
        process_detection(label_root / "det_val.json", image_root / "val", out_root / "val")

    elif args.task in ["drivable", "segmentation"]:
        label_dir = Path(args.raw_dir) / "labels" / args.task
        process_segmentation(image_root / "train", label_dir / "train", out_root / "train")
        process_segmentation(image_root / "val", label_dir / "val", out_root / "val")

if __name__ == "__main__":
    main()
