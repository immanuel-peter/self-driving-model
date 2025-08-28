import argparse
import json
import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.automoe import create_automoe_model
from dataloaders.carla_sequence_loader import CarlaSequenceDataset, carla_sequence_collate

def evaluate_model(model: nn.Module, 
                  loader: DataLoader, 
                  device: torch.device) -> Dict[str, float]:
    """Evaluate model performance"""
    model.eval()
    
    total_ade_l1 = 0.0
    total_fde_l1 = 0.0
    total_ade_euclid = 0.0
    total_fde_euclid = 0.0
    total_speed_loss = 0.0
    total_entropy = 0.0
    total_samples = 0
    
    expert_weights_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            pred = model(batch)
            
            bs = batch["waypoints"].size(0)

            # Compute metrics (L1 and Euclidean variants)
            ade_l1 = torch.nn.functional.l1_loss(pred["waypoints"], batch["waypoints"])  # already mean over dims
            fde_l1 = torch.nn.functional.l1_loss(pred["waypoints"][:, -1, :], batch["waypoints"][:, -1, :])

            # Euclidean ADE/FDE as commonly reported in trajectory prediction
            ade_euclid = (pred["waypoints"] - batch["waypoints"]).norm(dim=-1).mean()
            fde_euclid = (pred["waypoints"][:, -1] - batch["waypoints"][:, -1]).norm(dim=-1).mean()
            # Robust speed metric: prefer full sequence when available, else last step
            pred_spd = pred.get("speed_seq", pred.get("speed"))
            target_spd = batch["speed"]
            if (
                pred_spd is not None
                and isinstance(pred_spd, torch.Tensor)
                and pred_spd.dim() == 2
                and target_spd.dim() == 2
                and pred_spd.size(1) == target_spd.size(1)
            ):
                speed_loss = torch.nn.functional.l1_loss(pred_spd, target_spd)
            else:
                pred_last = pred.get("speed")
                if (
                    pred_last is not None
                    and isinstance(pred_last, torch.Tensor)
                    and pred_last.dim() == 2
                    and pred_last.size(1) == 1
                ):
                    speed_loss = torch.nn.functional.l1_loss(pred_last, target_spd[:, -1:].contiguous())
                else:
                    speed_loss = torch.zeros((), device=device)
            
            total_ade_l1 += float(ade_l1.item()) * bs
            total_fde_l1 += float(fde_l1.item()) * bs
            total_ade_euclid += float(ade_euclid.item()) * bs
            total_fde_euclid += float(fde_euclid.item()) * bs
            total_speed_loss += float(speed_loss.item()) * bs
            total_samples += bs
            
            # Collect expert weights
            expert_weights_list.append(pred["expert_weights"].cpu().numpy())
            # Entropy per-sample, averaged over batch
            w = pred["expert_weights"].clamp_min(1e-8)
            entropy = -(w * w.log()).sum(dim=1).mean()
            total_entropy += float(entropy.item()) * bs
    
    # Compute averages
    avg_ade_l1 = total_ade_l1 / total_samples
    avg_fde_l1 = total_fde_l1 / total_samples
    avg_ade_euclid = total_ade_euclid / total_samples
    avg_fde_euclid = total_fde_euclid / total_samples
    avg_speed_loss = total_speed_loss / total_samples
    avg_entropy = total_entropy / total_samples
    
    # Analyze expert usage
    expert_weights_array = np.concatenate(expert_weights_list, axis=0)
    expert_usage = expert_weights_array.mean(axis=0)
    expert_std = expert_weights_array.std(axis=0)
    
    return {
        'ade_l1': avg_ade_l1,
        'fde_l1': avg_fde_l1,
        'ade_euclid': avg_ade_euclid,
        'fde_euclid': avg_fde_euclid,
        'speed_loss': avg_speed_loss,
        'entropy': avg_entropy,
        'expert_usage': expert_usage,
        'expert_std': expert_std,
        'expert_weights': expert_weights_array
    }


def plot_expert_usage(expert_usage: np.ndarray, 
                     expert_std: np.ndarray, 
                     expert_names: List[str],
                     save_path: str):
    """Plot expert usage statistics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot of expert usage
    x = np.arange(len(expert_names))
    bars = ax1.bar(x, expert_usage, yerr=expert_std, capsize=5)
    ax1.set_xlabel('Expert')
    ax1.set_ylabel('Average Weight')
    ax1.set_title('Expert Usage Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(expert_names, rotation=45)
    
    # Add value labels on bars
    for bar, usage in zip(bars, expert_usage):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{usage:.3f}', ha='center', va='bottom')
    
    # Pie chart of expert usage
    ax2.pie(expert_usage, labels=expert_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Expert Usage Proportion')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Expert usage plot saved to {save_path}")


def plot_training_curves(log_dir: str, save_path: str):
    """Plot training curves from tensorboard logs"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        # Get scalar events
        scalar_tags = ea.Tags()['scalars']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, tag in enumerate(scalar_tags[:4]):  # Plot first 4 metrics
            events = ea.Scalars(tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            
            axes[i].plot(steps, values)
            axes[i].set_title(tag)
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel('Value')
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves plot saved to {save_path}")
        
    except ImportError:
        print("Tensorboard not available, skipping training curves plot")
    except Exception as e:
        print(f"Error plotting training curves: {e}")


def analyze_context_expert_correlation(model: nn.Module,
                                       loader: DataLoader,
                                       device: torch.device,
                                       save_path: str,
                                       expert_names: List[str],
                                       context_feature_names: List[str],
                                       use_logits: bool = False):
    """Correlate context features with gating in a space that makes sense:
       logits if available, else CLR-transformed weights. Includes guards and rank corr."""
    from scipy.stats import pearsonr, spearmanr

    model.eval()
    ctx_list, gate_list = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Analyzing context-expert correlation"):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # Build labeled context features (last-step where applicable)
            feats = []
            if isinstance(batch.get('speed', None), torch.Tensor):
                spd = batch['speed']
                feats.append(spd[:, -1:].contiguous() if spd.dim() == 2 else spd.view(spd.size(0), -1))
            for key in ('steering', 'throttle', 'brake'):
                t = batch.get(key, None)
                if isinstance(t, torch.Tensor):
                    feats.append(t[:, -1:].contiguous() if t.dim() == 2 else t.view(t.size(0), -1))
            if not feats:
                continue
            C = torch.cat(feats, dim=1)  # [B, C]

            # Get gating
            if use_logits and hasattr(model, "get_gating_logits"):
                G = model.get_gating_logits(batch)  # [B, E] pre-softmax; recommended
                G = G.float()
            else:
                W = model.get_expert_weights(batch).float()  # [B, E] probs
                # CLR transform to remove simplex artifact
                eps = 1e-8
                Wc = torch.clamp(W, eps, 1.0)
                logW = torch.log(Wc)
                G = logW - logW.mean(dim=1, keepdim=True)

            ctx_list.append(C.cpu().numpy())
            gate_list.append(G.cpu().numpy())

    if not ctx_list:
        print("No context features found for correlation analysis; skipping plot.")
        return

    C = np.concatenate(ctx_list, axis=0)  # [N, C]
    G = np.concatenate(gate_list, axis=0)  # [N, E]

    # Names
    c_names = (context_feature_names + [f"ctx_{i}" for i in range(999)])[:C.shape[1]]
    e_names = (expert_names + [f"E{i}" for i in range(999)])[:G.shape[1]]

    # Guard: drop near-constant columns to avoid exploding z-scores
    def good_cols(X, thr=1e-6):
        return np.where(X.std(axis=0) > thr)[0]

    c_keep = good_cols(C); g_keep = good_cols(G)
    C = C[:, c_keep]; G = G[:, g_keep]
    c_names = [c_names[i] for i in c_keep]; e_names = [e_names[j] for j in g_keep]

    # Compute Pearson and Spearman
    pear = np.zeros((C.shape[1], G.shape[1]), dtype=np.float32)
    spear = np.zeros_like(pear)
    for i in range(C.shape[1]):
        for j in range(G.shape[1]):
            p = pearsonr(C[:, i], G[:, j])[0]
            s = spearmanr(C[:, i], G[:, j])[0]
            pear[i, j] = 0.0 if np.isnan(p) else p
            spear[i, j] = 0.0 if np.isnan(s) else s

    # Plot Pearson heatmap (Spearman is often similar; save both)
    def plot_heat(M, title, path):
        fig, ax = plt.subplots(figsize=(1.6 * G.shape[1] + 3, 1.1 * C.shape[1] + 2))
        im = ax.imshow(M, cmap='RdBu_r', vmin=-0.8, vmax=0.8, aspect='auto')  # avoid false saturation
        ax.set_yticks(range(C.shape[1])); ax.set_yticklabels(c_names)
        ax.set_xticks(range(G.shape[1])); ax.set_xticklabels(e_names, rotation=45, ha='right')
        ax.set_xlabel('Experts' + (' (logits)' if use_logits else ' (CLR weights)'))
        ax.set_ylabel('Context Features')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"{title} saved to {path}")

    base = os.path.splitext(save_path)[0]
    plot_heat(pear, "Context vs Expert Correlation (Pearson)", base + "_pearson.png")
    plot_heat(spear, "Context vs Expert Correlation (Spearman)", base + "_spearman.png")

def main():
    parser = argparse.ArgumentParser(description="Evaluate AutoMoE gating network")
    parser.add_argument("--config", type=str, required=True, help="Path to training config file")
    parser.add_argument("--model_config", type=str, default="models/configs/automoe/model_config.json", help="Path to model config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Path to CARLA data")
    parser.add_argument("--output_dir", type=str, default="eval/results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val", help="Dataset split")
    
    args = parser.parse_args()
    
    # Load configurations
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = json.load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_automoe_model(model_config, device)
    
    # Load checkpoint with DDP compatibility (strip 'module.' prefixes)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Strict load failed: {e}\nRetrying with strict=False...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded with relaxed matching. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Create dataset and loader
    # Use model_config/policy horizon to match number of waypoints expected
    horizon = int(model_config.get('policy', {}).get('num_waypoints', 10))
    test_dataset = CarlaSequenceDataset(
        root_dir=args.data_root,
        split=args.split,
        past=0,
        horizon=horizon,
        stride=1,
        include_context=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=carla_sequence_collate,
        pin_memory=True
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"ADE (L1): {results['ade_l1']:.4f}")
    print(f"FDE (L1): {results['fde_l1']:.4f}")
    print(f"ADE (Euclid): {results['ade_euclid']:.4f}")
    print(f"FDE (Euclid): {results['fde_euclid']:.4f}")
    print(f"Speed Loss: {results['speed_loss']:.4f}")
    print(f"Gating Entropy: {results['entropy']:.4f}")
    print("\nExpert Usage:")
    expert_names = ['Detection', 'Segmentation', 'Drivable', 'nuScenes']
    for i, (name, usage, std) in enumerate(zip(expert_names, results['expert_usage'], results['expert_std'])):
        print(f"  {name}: {usage:.3f} ± {std:.3f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'ade_l1': results['ade_l1'],
            'fde_l1': results['fde_l1'],
            'ade_euclid': results['ade_euclid'],
            'fde_euclid': results['fde_euclid'],
            'speed_loss': results['speed_loss'],
            'entropy': results['entropy'],
            'expert_usage': results['expert_usage'].tolist(),
            'expert_std': results['expert_std'].tolist()
        }, f, indent=4)
    print(f"\nResults saved to {results_file}")
    
    # Create plots
    print("\nCreating plots...")
    
    # Expert usage plot
    usage_plot_path = os.path.join(args.output_dir, 'expert_usage.png')
    plot_expert_usage(results['expert_usage'], results['expert_std'], expert_names, usage_plot_path)
    
    # Training curves plot
    log_dir = f"models/runs/gating_network_{config.get('run_name', 'default')}"
    if os.path.exists(log_dir):
        curves_plot_path = os.path.join(args.output_dir, 'training_curves.png')
        plot_training_curves(log_dir, curves_plot_path)
    
    # Context-expert correlation plot (labeled)
    correlation_plot_path = os.path.join(args.output_dir, 'context_expert_correlation.png')
    expert_names = [
        'Detection',
        'Segmentation',
        'Drivable',
        'nuScenes',
    ]
    # Context features we expect to extract (order must match extraction above)
    context_feature_names = ['speed_last', 'steering_last', 'throttle_last', 'brake_last']
    analyze_context_expert_correlation(
        model, test_loader, device, correlation_plot_path,
        expert_names=expert_names,
        context_feature_names=context_feature_names,
    )

    # Sanity check: expert weights sum to ~1
    sums = results['expert_weights'].sum(axis=1)
    print(f"Expert weight sums: min={sums.min():.4f}, max={sums.max():.4f}, mean={sums.mean():.4f}")
    
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

