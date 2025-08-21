import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataloaders.carla_sequence_loader import get_carla_sequence_loader
from models.policy.trajectory_head import TrajectoryPolicy


def compute_losses(pred: Dict[str, torch.Tensor], target_wp: torch.Tensor, target_spd: torch.Tensor) -> Dict[str, torch.Tensor]:
    # ADE over horizon
    ade = torch.nn.functional.l1_loss(pred["waypoints"], target_wp)
    # FDE on final step
    fde = torch.nn.functional.l1_loss(pred["waypoints"][:, -1, :], target_wp[:, -1, :])
    # Speed L1
    l_spd = torch.nn.functional.l1_loss(pred["speed"], target_spd)
    # Smoothness on waypoint deltas
    pred_deltas = pred["waypoints"][:, 1:, :] - pred["waypoints"][:, :-1, :]
    l_smooth = torch.nn.functional.l1_loss(pred_deltas[:, 1:, :], pred_deltas[:, :-1, :])

    loss = ade + 2.0 * fde + 0.2 * l_spd + 0.1 * l_smooth
    return {"loss": loss, "ade": ade, "fde": fde, "speed": l_spd, "smooth": l_smooth}


def train_one_epoch(model: TrajectoryPolicy, loader, optimizer, device: torch.device):
    model.train()
    pbar = tqdm(loader, desc="train")
    total = 0.0
    for batch in pbar:
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
        pbar.set_postfix({"loss": f"{metrics['loss'].item():.3f}", "ade": f"{metrics['ade'].item():.3f}"})
    return total / max(1, len(loader))


@torch.no_grad()
def validate(model: TrajectoryPolicy, loader, device: torch.device):
    model.eval()
    pbar = tqdm(loader, desc="val")
    total = 0.0
    for batch in pbar:
        images = batch["image"].to(device)
        target_wp = batch["waypoints"].to(device)
        target_spd = batch["speed"].to(device)
        context = batch.get("context", None)
        if context is not None:
            context = context.to(device)
        pred = model(images, context)
        metrics = compute_losses(pred, target_wp, target_spd)
        total += float(metrics["loss"].item())
        pbar.set_postfix({"loss": f"{metrics['loss'].item():.3f}", "ade": f"{metrics['ade'].item():.3f}"})
    return total / max(1, len(loader))


def train(model: TrajectoryPolicy,
          train_loader,
          val_loader,
          optimizer: optim.Optimizer,
          device: torch.device,
          epochs: int) -> float:
    best_val = float("inf")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        best_val = min(best_val, val_loss)
        print(f"epoch {epoch+1}/{epochs}: train {train_loss:.4f} | val {val_loss:.4f} | best {best_val:.4f}")
    return best_val


def main():
    parser = argparse.ArgumentParser(description="Train CARLA trajectory policy (waypoints + speed)")
    parser.add_argument("--data_root", type=str, default="datasets/carla/preprocessed")
    parser.add_argument("--epochs", type=int, default=0, help="0 means dry-run (no training)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no_context", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_carla_sequence_loader(
        split="train", root_dir=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        horizon=args.horizon, include_context=(not args.no_context)
    )
    val_loader = get_carla_sequence_loader(
        split="val", root_dir=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        horizon=args.horizon, include_context=(not args.no_context), shuffle=False
    )

    context_dim = 0
    # Peek one batch to infer context dim if present (safe even for dry-run)
    sample_batch = next(iter(train_loader))
    if "context" in sample_batch:
        context_dim = int(sample_batch["context"].shape[1])

    model = TrajectoryPolicy(horizon=args.horizon, context_dim=context_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.epochs <= 0:
        # Dry run: one forward pass over a small batch from train/val
        with torch.no_grad():
            images = sample_batch["image"].to(device)
            context = sample_batch.get("context", None)
            if context is not None:
                context = context.to(device)
            out = model(images, context)
            print({k: tuple(v.shape) for k, v in out.items()})
        return

    best_val = train(model, train_loader, val_loader, optimizer, device, args.epochs)
    print(f"training complete. best val {best_val:.4f}")


if __name__ == "__main__":
    main()