import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_CARLA_PREPROCESSED_ROOT = "datasets/carla/preprocessed"

def _list_run_dirs(split_dir: Path) -> List[Path]:
    return sorted([d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])

def _list_pt_files(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("*.pt"))

def _deg2rad(degrees: float) -> float:
    return degrees * math.pi / 180.0

def _world_to_ego_xy(p_world_xy: torch.Tensor,
                     ego_origin_xy: torch.Tensor,
                     ego_yaw_deg: float) -> torch.Tensor:
    """Convert a point from world XY to ego XY at time t.

    Args:
        p_world_xy: tensor [2] (x, y) in world coordinates
        ego_origin_xy: tensor [2] ego position at t in world XY
        ego_yaw_deg: float yaw in degrees at t (CARLA convention)

    Returns:
        tensor [2] point in ego frame (x right, y forward)
    """
    # Translate
    delta = p_world_xy - ego_origin_xy
    # Rotate by -yaw to align world to ego heading
    yaw = _deg2rad(ego_yaw_deg)
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    rot = torch.tensor([[cos_yaw, -sin_yaw],
                        [sin_yaw,  cos_yaw]], dtype=torch.float32)
    p_ego = rot @ delta
    # Adopt convention: x right, y forward
    return p_ego


class CarlaSequenceDataset(Dataset):
    def __init__(self,
                 split: str = "train",
                 root_dir: str = DEFAULT_CARLA_PREPROCESSED_ROOT,
                 past: int = 0,
                 horizon: int = 8,
                 stride: int = 1,
                 include_context: bool = True):
        """Sequence dataset over preprocessed CARLA .pt files to build waypoint targets.

        Window contains [t] (current) and future frames [t+1, ..., t+horizon].

        Args:
            split: "train" or "val"
            root_dir: path to preprocessed root with split subfolders
            past: number of past frames before t (reserved for future use). Currently unused; windows include only t and future.
            horizon: number of future steps
            stride: step between consecutive windows
            include_context: include weather and traffic context if present
        """
        super().__init__()
        self.split = split
        self.root = Path(root_dir) / split
        self.past = max(0, int(past))
        self.horizon = int(horizon)
        self.stride = max(1, int(stride))
        self.include_context = include_context

        if not self.root.exists():
            raise FileNotFoundError(f"Split directory not found: {self.root}")

        self.runs: List[Path] = _list_run_dirs(self.root)
        if not self.runs:
            raise RuntimeError(f"No run directories found under {self.root}")

        # Pre-index windows as (run_dir, start_index t), where window covers [t, t+1..t+horizon]
        self.index: List[Tuple[Path, int]] = []
        for run_dir in self.runs:
            pt_files = _list_pt_files(run_dir)
            # Need at least (1 + horizon) frames starting at t
            max_start = len(pt_files) - (1 + self.horizon)
            if max_start < 0:
                continue
            for t in range(0, max_start + 1, self.stride):
                self.index.append((run_dir, t))

        if not self.index:
            raise RuntimeError(f"No valid windows found under {self.root}")

    def __len__(self) -> int:
        return len(self.index)

    def _load_frame(self, run_dir: Path, frame_idx: int) -> Dict[str, Any]:
        pt_path = _list_pt_files(run_dir)[frame_idx]
        sample: Dict[str, Any] = torch.load(pt_path, weights_only=False)
        return sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        run_dir, t = self.index[idx]
        # Current frame at t and future frames t+1..t+horizon
        current = self._load_frame(run_dir, t)
        futures = [self._load_frame(run_dir, t + k) for k in range(1, self.horizon + 1)]

        # Inputs
        image_t: torch.Tensor = current["image"]  # [3, 256, 256]

        # Ego pose at t
        loc_t = current["vehicle_state"]["location"].to(torch.float32)  # [x, y, z]
        rot_t = current["vehicle_state"]["rotation"].to(torch.float32)   # [pitch, yaw, roll]
        yaw_t_deg: float = float(rot_t[1].item())

        # Build waypoints (ego frame) and speed profile from futures
        waypoints_xy: List[torch.Tensor] = []
        speeds: List[float] = []
        origin_xy = loc_t[:2].to(torch.float32)

        for f in futures:
            loc_f = f["vehicle_state"]["location"].to(torch.float32)
            p_world_xy = loc_f[:2]
            p_ego = _world_to_ego_xy(p_world_xy, origin_xy, yaw_t_deg)
            waypoints_xy.append(p_ego)
            speeds.append(float(f["vehicle_state"]["speed_kmh"].item()))

        waypoints = torch.stack(waypoints_xy, dim=0)  # [H, 2]
        speed_profile = torch.tensor(speeds, dtype=torch.float32)  # [H]

        # Optional compact context vector if present in preprocessed sample
        context_tensor: Optional[torch.Tensor] = None
        if self.include_context and ("context" in current):
            ctx_parts: List[torch.Tensor] = []
            if isinstance(current["context"], dict):
                weather = current["context"].get("weather", None)
                traffic = current["context"].get("traffic_density", None)
                if isinstance(weather, torch.Tensor):
                    ctx_parts.append(weather.to(torch.float32).flatten())
                if isinstance(traffic, torch.Tensor):
                    ctx_parts.append(traffic.to(torch.float32).flatten())
            if ctx_parts:
                context_tensor = torch.cat(ctx_parts, dim=0)

        out: Dict[str, Any] = {
            "image": image_t,
            "waypoints": waypoints,
            "speed": speed_profile,
            "meta": {
                "run_id": current.get("meta", {}).get("run_id", run_dir.name),
                "frame_id": int(current.get("meta", {}).get("frame_id", t)),
            }
        }
        if context_tensor is not None:
            out["context"] = context_tensor
        return out


def carla_sequence_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    waypoints = torch.stack([b["waypoints"] for b in batch], dim=0)
    speeds = torch.stack([b["speed"] for b in batch], dim=0)
    contexts: Optional[torch.Tensor] = None
    if "context" in batch[0]:
        contexts = torch.stack([b["context"] for b in batch], dim=0)
    metas = [b["meta"] for b in batch]

    out: Dict[str, Any] = {
        "image": images,
        "waypoints": waypoints,
        "speed": speeds,
        "meta": metas,
    }
    if contexts is not None:
        out["context"] = contexts
    return out


def get_carla_sequence_loader(split: str = "train",
                              root_dir: str = DEFAULT_CARLA_PREPROCESSED_ROOT,
                              batch_size: int = 32,
                              num_workers: int = 4,
                              past: int = 0,
                              horizon: int = 8,
                              stride: int = 1,
                              include_context: bool = True,
                              shuffle: Optional[bool] = None) -> DataLoader:
    if shuffle is None:
        shuffle = (split == "train")
    dataset = CarlaSequenceDataset(
        split=split,
        root_dir=root_dir,
        past=past,
        horizon=horizon,
        stride=stride,
        include_context=include_context,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        collate_fn=carla_sequence_collate,
    )