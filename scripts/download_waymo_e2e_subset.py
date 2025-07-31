import argparse
import subprocess
import os
import random

E2E_BUCKET = "gs://waymo_open_dataset_end_to_end_camera_v_1_0_0"
DEFAULT_OUTDIR = "./datasets/waymo/e2e/raw"
GSUTIL_PARALLEL_THREAD_COUNT = 16
GSUTIL_PARALLEL_PROCESS_COUNT = 4

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    return result.stdout.strip().splitlines()

def download_files(file_list, destination):
    if not file_list:
        print("No files to download.")
        return

    os.makedirs(destination, exist_ok=True)
    for file in file_list:
        command = f"gsutil -o GSUtil:parallel_thread_count={GSUTIL_PARALLEL_THREAD_COUNT} -o GSUtil:parallel_process_count={GSUTIL_PARALLEL_PROCESS_COUNT} cp {file} '{destination}/'"
        print(f"Downloading {file} to {destination}")
        try:
            subprocess.run(command, shell=True, check=True)
            print("Download complete.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred during download: {e.stderr}")

def main():
    parser = argparse.ArgumentParser(description="Download a subset of Waymo E2E records from GCS")
    parser.add_argument("--bucket", default=E2E_BUCKET, help="GCS bucket path")
    parser.add_argument("--train", type=int, default=80, help="Number of training records to download")
    parser.add_argument("--val", type=int, default=20, help="Number of validation records to download")
    parser.add_argument("--random", action="store_true", help="Shuffle before sampling")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")

    args = parser.parse_args()

    # List all files in the bucket
    print("Listing all files in GCS bucket...")
    all_files = run_cmd(f"gsutil ls {args.bucket}")

    train_files = [f for f in all_files if "training_" in f and ".tfrecord" in f]
    val_files = [f for f in all_files if "val_" in f and ".tfrecord" in f]

    if args.random:
        random.shuffle(train_files)
        random.shuffle(val_files)

    selected_train = train_files[:args.train]
    selected_val = val_files[:args.val]
    
    val_seq_map = f"{E2E_BUCKET}/val_sequence_name_to_scenario_cluster.json"
    if val_seq_map not in selected_val:
        selected_val.append(val_seq_map)

    print(f"Downloading {len(selected_train)} training files...")
    download_files(selected_train, os.path.join(args.outdir, "train"))

    print(f"Downloading {len(selected_val)} validation files...")
    download_files(selected_val, os.path.join(args.outdir, "val"))

    print("✅ Done.")

if __name__ == "__main__":
    main()
