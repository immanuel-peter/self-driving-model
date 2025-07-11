import os
import torch
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def convert_detection_pt_to_df(pt_files):
    data = []
    for pt_path in tqdm(pt_files, desc="Processing detection"):
        sample = torch.load(pt_path)
        for box, label in zip(sample["bboxes"], sample["labels"]):
            data.append({
                "image_path": sample["image_path"],
                "bbox": box.tolist(),           # [x1, y1, x2, y2]
                "label": int(label),            # int class ID
                "scene": sample["meta"]["scene"],
                "timeofday": sample["meta"]["timeofday"],
                "weather": sample["meta"]["weather"]
            })
    return pd.DataFrame(data)

def convert_mask_pt_to_df(pt_files, task):
    data = []
    for pt_path in tqdm(pt_files, desc=f"Processing {task}"):
        sample = torch.load(pt_path)
        data.append({
            "image_path": sample["image_path"],
            "mask_path": sample["mask_path"]
        })
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["detection", "drivable", "segmentation"])
    parser.add_argument("--split", type=str, required=True, choices=["train", "val"])
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    pt_files = sorted(Path(args.input_dir).glob("*.pt"))

    if args.task == "detection":
        df = convert_detection_pt_to_df(pt_files)
    else:
        df = convert_mask_pt_to_df(pt_files, args.task)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    df.to_parquet(args.out_path, index=False)
    print(f"Saved to {args.out_path}")

if __name__ == "__main__":
    main()
