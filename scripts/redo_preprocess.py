import shutil
from pathlib import Path
import argparse

def parse_task(task_str):
    if ":" in task_str:
        dataset, subtask = task_str.split(":", 1)
    else:
        dataset, subtask = task_str, None
    return dataset.lower(), subtask

def clear_preprocessed(dataset, subtask):
    base_path = Path("datasets") / dataset / "preprocessed"
    target_path = base_path / subtask if subtask else base_path

    if target_path.exists():
        print(f"🧹 Deleting existing: {target_path}")
        shutil.rmtree(target_path)
    else:
        print(f"✅ No existing data at {target_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        help="Format: dataset[:subtask], e.g. waymo:motion, waymo:perception, waymo:e2e, bdd100k, nuscenes, carla, cosmos")
    args = parser.parse_args()

    dataset, subtask = parse_task(args.task)
    clear_preprocessed(dataset, subtask)

if __name__ == "__main__":
    main()
