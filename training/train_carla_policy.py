import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloaders.carla_sequence_loader import CarlaSequenceDataset, carla_sequence_collate
from models.policy.trajectory_head import TrajectoryPolicy


def compute_losses(pred: Dict[str, torch.Tensor], target_wp: torch.Tensor, target_spd: torch.Tensor) -> Dict[str, torch.Tensor]:
    # ADE over horizon
    ade = F.l1_loss(pred["waypoints"], target_wp)
    # FDE on final step
    fde = F.l1_loss(pred["waypoints"][:, -1, :], target_wp[:, -1, :])
    # Speed L1
    l_spd = F.l1_loss(pred["speed"], target_spd)
    # Smoothness on waypoint deltas
    pred_deltas = pred["waypoints"][:, 1:, :] - pred["waypoints"][:, :-1, :]
    l_smooth = F.l1_loss(pred_deltas[:, 1:, :], pred_deltas[:, :-1, :])

    loss = ade + 2.0 * fde + 0.2 * l_spd + 0.1 * l_smooth
    return {"loss": loss, "ade": ade, "fde": fde, "speed": l_spd, "smooth": l_smooth}


def train_one_epoch(model: TrajectoryPolicy, loader, optimizer, device: torch.device, epoch_idx: int, epochs: int, rank: int):
    model.train()
    desc = f"train | epoch {epoch_idx+1}/{epochs}"
    it = loader if rank != 0 else tqdm(loader, desc=desc)
    total = 0.0
    for batch in it:
        images = batch["image"].to(device)
        target_wp = batch["waypoints"].to(device)
        target_spd = batch["speed"].to(device)
        context = batch.get("context", None)
        if context is not None:
            context = context.to(device)

        optimizer.zero_grad()
        pred = model(images, context)
        metrics = compute_losses(pred, target_wp, target_spd)
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(metrics["loss"].item())
        if rank == 0:
            it.set_postfix({"loss": f"{metrics['loss'].item():.3f}", "ade": f"{metrics['ade'].item():.3f}"})
    return total / max(1, len(loader))


@torch.no_grad()
def validate(model: TrajectoryPolicy, loader, device: torch.device, epoch_idx: int, epochs: int, rank: int, world_size: int):
    model.eval()
    desc = f"val | epoch {epoch_idx+1}/{epochs}"
    it = loader if rank != 0 else tqdm(loader, desc=desc)
    total = 0.0
    count = 0
    for batch in it:
        images = batch["image"].to(device)
        target_wp = batch["waypoints"].to(device)
        target_spd = batch["speed"].to(device)
        context = batch.get("context", None)
        if context is not None:
            context = context.to(device)
        pred = model(images, context)
        metrics = compute_losses(pred, target_wp, target_spd)
        total += float(metrics["loss"].item())
        count += 1
        if rank == 0:
            it.set_postfix({"loss": f"{metrics['loss'].item():.3f}", "ade": f"{metrics['ade'].item():.3f}"})
    # Reduce across ranks
    if dist.is_initialized():
        t = torch.tensor([total, count], dtype=torch.float32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total, count = float(t[0].item()), int(t[1].item())
    return total / max(1, count)


def train(model: TrajectoryPolicy,
          train_loader,
          val_loader,
          optimizer: optim.Optimizer,
          device: torch.device,
          epochs: int,
          rank: int,
          world_size: int,
          run_name: str,
          ckpt_root: str) -> float:
    best_val = float("inf")
    start = time.time()
    ckpt_dir = Path(ckpt_root) / run_name
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, epochs, rank)
        val_loss = validate(model, val_loader, device, epoch, epochs, rank, world_size)
        if rank == 0:
            if val_loss < best_val:
                best_val = val_loss
                # Save best checkpoint (DDP-safe: unwrap if needed)
                to_save = model.module if isinstance(model, DDP) else model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val,
                    'horizon': getattr(to_save, 'horizon', None),
                }, ckpt_dir / 'best.pth')
            print(f"epoch {epoch+1}/{epochs}: train {train_loss:.4f} | val {val_loss:.4f} | best {best_val:.4f}")
    if rank == 0:
        print(f"total_time_s={time.time()-start:.1f}")
        # Save final checkpoint as well
        to_save = model.module if isinstance(model, DDP) else model
        torch.save({
            'epoch': epochs,
            'model_state_dict': to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val,
            'horizon': getattr(to_save, 'horizon', None),
        }, ckpt_dir / 'last.pth')
    return best_val


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29600')
        dist.init_process_group(backend='nccl', world_size=world_size, rank=int(os.environ.get('RANK', local_rank)))
        torch.cuda.set_device(local_rank)
    return world_size, local_rank


def main():
    parser = argparse.ArgumentParser(description="Train CARLA trajectory policy (waypoints + speed)")
    parser.add_argument("--data_root", type=str, default="datasets/carla/preprocessed")
    parser.add_argument("--epochs", type=int, default=0, help="0 means dry-run (no training)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no_context", action="store_true")
    parser.add_argument("--run_name", type=str, default="carla_policy_ddp")
    parser.add_argument("--ckpt_dir", type=str, default="models/checkpoints/carla_policy")
    args = parser.parse_args()

    world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Build datasets and DDP samplers
    train_ds = CarlaSequenceDataset(split="train", root_dir=args.data_root, horizon=args.horizon, include_context=(not args.no_context))
    val_ds = CarlaSequenceDataset(split="val", root_dir=args.data_root, horizon=args.horizon, include_context=(not args.no_context))
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True, drop_last=True, collate_fn=carla_sequence_collate, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False, collate_fn=carla_sequence_collate, persistent_workers=True, prefetch_factor=4)

    # Infer context dim from dataset directly
    context_dim = 0
    sample0 = train_ds[0]
    if "context" in sample0:
        context_dim = int(sample0["context"].shape[0])

    model = TrajectoryPolicy(horizon=args.horizon, context_dim=context_dim).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None, output_device=local_rank if device.type == 'cuda' else None, find_unused_parameters=False)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.epochs <= 0:
        # Dry run: one forward pass over a small batch from train/val
        with torch.no_grad():
            sb = train_ds[0]
            images = sb["image"].unsqueeze(0).to(device)
            context = sb.get("context", None)
            if context is not None:
                context = context.unsqueeze(0).to(device)
            out = model(images, context)
            if local_rank == 0:
                print({k: tuple(v.shape) for k, v in out.items()})
        return

    # Save run config (rank 0)
    if local_rank == 0:
        cfg_dir = Path("models/configs/carla_policy") / args.run_name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

    best_val = train(model, train_loader, val_loader, optimizer, device, args.epochs, local_rank, world_size, args.run_name, args.ckpt_dir)
    if local_rank == 0:
        print(f"training complete. best val {best_val:.4f}")


if __name__ == "__main__":
    main()