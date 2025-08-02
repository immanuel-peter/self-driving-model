"""
Evaluation script for the Mixture of Experts (MoE) model.
This implements Step 5 of the AutoMoE roadmap: Evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

# Import models
from models.gating_network import GatingNetwork, MoEModel
from models.experts import (
    BDDDetectionExpert, 
    BDDSegmentationExpert, 
    BDDDrivableExpert, 
    NuScenesExpert
)

# Import dataloaders
import dataloaders


class MoEEvaluator:
    def __init__(self, moe_model, val_loader, device, save_dir):
        self.moe_model = moe_model.to(device)
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.expert_usage_stats = defaultdict(list)
        self.predictions = []
        self.ground_truths = []
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def load_model_checkpoint(self, checkpoint_path):
        """Load trained MoE model checkpoint"""
        print(f"Loading MoE model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load gating network
        self.moe_model.gating_network.load_state_dict(checkpoint['gating_network_state_dict'])
        
        # Load expert projections
        self.moe_model.expert_projections.load_state_dict(checkpoint['expert_projections_state_dict'])
        
        print("MoE model loaded successfully!")
    
    def load_expert_checkpoints(self, expert_checkpoint_paths):
        """Load expert checkpoints"""
        print("Loading expert checkpoints...")
        
        for expert_name, checkpoint_path in expert_checkpoint_paths.items():
            if expert_name in self.moe_model.experts:
                print(f"Loading {expert_name} from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.moe_model.experts[expert_name].load_state_dict(checkpoint['model_state_dict'])
    
    def extract_control_target(self, sample):
        """Extract control target from sample"""
        if 'steering' in sample and 'throttle' in sample:
            steering = sample['steering']
            throttle = sample['throttle']
            
            if isinstance(steering, (list, tuple)):
                steering = torch.tensor(steering, device=self.device)
            if isinstance(throttle, (list, tuple)):
                throttle = torch.tensor(throttle, device=self.device)
            
            if len(steering.shape) == 0:
                steering = steering.unsqueeze(0)
            if len(throttle.shape) == 0:
                throttle = throttle.unsqueeze(0)
            
            control_target = torch.stack([steering, throttle], dim=1)
            return control_target
        return None
    
    def extract_context_data(self, sample):
        """Extract context data from sample"""
        weather_data = {}
        traffic_data = {}
        
        # Extract weather context
        if 'weather' in sample:
            weather = sample['weather']
            weather_data = {
                'category': weather.get('category', torch.zeros(1, dtype=torch.long, device=self.device)),
                'visibility': weather.get('visibility', torch.ones(1, device=self.device)),
                'wetness': weather.get('wetness', torch.zeros(1, device=self.device)),
                'cloudiness': weather.get('cloudiness', torch.zeros(1, device=self.device))
            }
        
        # Extract traffic context
        if 'traffic' in sample:
            traffic = sample['traffic']
            traffic_data = {
                'vehicle_count': traffic.get('vehicle_count', torch.zeros(1, device=self.device)),
                'pedestrian_count': traffic.get('pedestrian_count', torch.zeros(1, device=self.device)),
                'traffic_light_state': traffic.get('traffic_light_state', torch.zeros(1, device=self.device)),
                'speed_limit': traffic.get('speed_limit', torch.ones(1, device=self.device) * 50.0)
            }
        
        return weather_data if weather_data else None, traffic_data if traffic_data else None
    
    def evaluate_single_expert(self, expert_name, expert_model, batch):
        """Evaluate a single expert on the batch"""
        try:
            image = batch.get('image', batch.get('rgb'))
            
            if expert_name == 'nuscenes':
                nuscenes_input = {
                    'image': image,
                    'lidar': batch.get('lidar', [torch.zeros(100, 3, device=self.device) for _ in range(image.size(0))])
                }
                expert_output = expert_model(nuscenes_input)
            else:
                expert_output = expert_model(image)
            
            # Project to control space
            if isinstance(expert_output, dict):
                expert_features = torch.cat([
                    expert_output['class_logits'].mean(dim=[2, 3]),
                    expert_output['bbox_deltas'].mean(dim=[2, 3])
                ], dim=1)
            elif len(expert_output.shape) == 4:
                expert_features = expert_output.mean(dim=[2, 3])
            else:
                expert_features = expert_output
            
            control_pred = self.moe_model.expert_projections[expert_name](expert_features)
            return control_pred
            
        except Exception as e:
            print(f"Error evaluating {expert_name}: {e}")
            return None
    
    def evaluate(self):
        """Main evaluation loop"""
        print("Starting MoE model evaluation...")
        
        self.moe_model.eval()
        all_moe_predictions = []
        all_targets = []
        all_expert_weights = []
        all_expert_predictions = {name: [] for name in self.moe_model.experts.keys()}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating")):
                try:
                    # Handle batch format
                    if isinstance(batch, list):
                        batch = self.collate_batch(batch)
                    
                    batch = self.move_to_device(batch)
                    
                    # Extract inputs
                    image = batch.get('image', batch.get('rgb'))
                    lidar = batch.get('lidar', None)
                    weather_data, traffic_data = self.extract_context_data(batch)
                    control_target = self.extract_control_target(batch)
                    
                    if control_target is None:
                        continue
                    
                    # MoE model prediction
                    moe_output = self.moe_model(image, lidar, weather_data, traffic_data)
                    moe_prediction = moe_output['final_output']
                    expert_weights = moe_output['expert_weights']
                    
                    # Individual expert predictions
                    expert_predictions = {}
                    for expert_name, expert_model in self.moe_model.experts.items():
                        pred = self.evaluate_single_expert(expert_name, expert_model, batch)
                        if pred is not None:
                            expert_predictions[expert_name] = pred
                    
                    # Store results
                    all_moe_predictions.append(moe_prediction.cpu())
                    all_targets.append(control_target.cpu())
                    all_expert_weights.append(expert_weights.cpu())
                    
                    for expert_name, pred in expert_predictions.items():
                        all_expert_predictions[expert_name].append(pred.cpu())
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
        
        # Convert to tensors
        all_moe_predictions = torch.cat(all_moe_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_expert_weights = torch.cat(all_expert_weights, dim=0)
        
        for expert_name in all_expert_predictions:
            if all_expert_predictions[expert_name]:
                all_expert_predictions[expert_name] = torch.cat(all_expert_predictions[expert_name], dim=0)
        
        # Compute metrics
        metrics = self.compute_metrics(all_moe_predictions, all_targets, all_expert_weights, all_expert_predictions)
        
        # Generate visualizations
        self.create_visualizations(all_moe_predictions, all_targets, all_expert_weights, all_expert_predictions)
        
        # Save results
        self.save_results(metrics, all_moe_predictions, all_targets, all_expert_weights)
        
        return metrics
    
    def compute_metrics(self, moe_predictions, targets, expert_weights, expert_predictions):
        """Compute evaluation metrics"""
        metrics = {}
        
        # MoE model metrics
        moe_mse = self.mse_loss(moe_predictions, targets).item()
        moe_mae = self.mae_loss(moe_predictions, targets).item()
        
        # Per-dimension metrics
        steering_mse = self.mse_loss(moe_predictions[:, 0], targets[:, 0]).item()
        throttle_mse = self.mse_loss(moe_predictions[:, 1], targets[:, 1]).item()
        steering_mae = self.mae_loss(moe_predictions[:, 0], targets[:, 0]).item()
        throttle_mae = self.mae_loss(moe_predictions[:, 1], targets[:, 1]).item()
        
        metrics['moe'] = {
            'mse': moe_mse,
            'mae': moe_mae,
            'steering_mse': steering_mse,
            'throttle_mse': throttle_mse,
            'steering_mae': steering_mae,
            'throttle_mae': throttle_mae,
            'rmse': np.sqrt(moe_mse),
            'steering_rmse': np.sqrt(steering_mse),
            'throttle_rmse': np.sqrt(throttle_mse)
        }
        
        # Individual expert metrics
        for expert_name, predictions in expert_predictions.items():
            if len(predictions) > 0:
                expert_mse = self.mse_loss(predictions, targets).item()
                expert_mae = self.mae_loss(predictions, targets).item()
                
                metrics[f'expert_{expert_name}'] = {
                    'mse': expert_mse,
                    'mae': expert_mae,
                    'rmse': np.sqrt(expert_mse)
                }
        
        # Expert usage statistics
        expert_usage = expert_weights.mean(dim=0)  # Average weight per expert
        expert_selection_counts = expert_weights.argmax(dim=1).bincount(minlength=expert_weights.size(1))
        expert_selection_rates = expert_selection_counts.float() / expert_weights.size(0)
        
        metrics['expert_usage'] = {
            'average_weights': expert_usage.tolist(),
            'selection_counts': expert_selection_counts.tolist(),
            'selection_rates': expert_selection_rates.tolist(),
            'entropy': self.compute_entropy(expert_weights),
            'gini_coefficient': self.compute_gini_coefficient(expert_usage)
        }
        
        print("\\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"MoE Model MSE: {moe_mse:.6f}")
        print(f"MoE Model MAE: {moe_mae:.6f}")
        print(f"MoE Model RMSE: {np.sqrt(moe_mse):.6f}")
        print(f"Steering MSE: {steering_mse:.6f}")
        print(f"Throttle MSE: {throttle_mse:.6f}")
        print("\\nExpert Usage:")
        for i, (weight, rate) in enumerate(zip(expert_usage, expert_selection_rates)):
            expert_name = self.moe_model.gating_network.expert_names[i]
            print(f"  {expert_name}: {weight:.3f} avg weight, {rate:.3f} selection rate")
        print("="*50)
        
        return metrics
    
    def compute_entropy(self, expert_weights):
        """Compute entropy of expert selection distribution"""
        avg_weights = expert_weights.mean(dim=0)
        entropy = -(avg_weights * torch.log(avg_weights + 1e-8)).sum().item()
        return entropy
    
    def compute_gini_coefficient(self, weights):
        """Compute Gini coefficient for expert usage inequality"""
        weights_sorted = torch.sort(weights)[0]
        n = len(weights_sorted)
        index = torch.arange(1, n + 1, dtype=torch.float)
        gini = (2 * (index * weights_sorted).sum()) / (n * weights_sorted.sum()) - (n + 1) / n
        return gini.item()
    
    def create_visualizations(self, moe_predictions, targets, expert_weights, expert_predictions):
        """Create visualization plots"""
        print("Creating visualizations...")
        
        # 1. Prediction vs Ground Truth scatter plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Steering
        axes[0].scatter(targets[:, 0], moe_predictions[:, 0], alpha=0.6)
        axes[0].plot([-1, 1], [-1, 1], 'r--', label='Perfect prediction')
        axes[0].set_xlabel('Ground Truth Steering')
        axes[0].set_ylabel('Predicted Steering')
        axes[0].set_title('Steering Prediction vs Ground Truth')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Throttle
        axes[1].scatter(targets[:, 1], moe_predictions[:, 1], alpha=0.6)
        axes[1].plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
        axes[1].set_xlabel('Ground Truth Throttle')
        axes[1].set_ylabel('Predicted Throttle')
        axes[1].set_title('Throttle Prediction vs Ground Truth')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'prediction_vs_ground_truth.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Expert weight distribution
        plt.figure(figsize=(10, 6))
        expert_names = self.moe_model.gating_network.expert_names
        avg_weights = expert_weights.mean(dim=0)
        
        bars = plt.bar(expert_names, avg_weights)
        plt.title('Average Expert Weights')
        plt.ylabel('Average Weight')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, weight in zip(bars, avg_weights):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{weight:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'expert_weights.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Expert weight heatmap over time
        if expert_weights.size(0) > 100:  # Only if we have enough samples
            sample_indices = torch.linspace(0, expert_weights.size(0)-1, 100).long()
            weights_subset = expert_weights[sample_indices].numpy()
            
            plt.figure(figsize=(12, 6))
            sns.heatmap(weights_subset.T, 
                       yticklabels=expert_names,
                       cmap='Blues',
                       cbar_kws={'label': 'Expert Weight'})
            plt.title('Expert Weights Over Time')
            plt.xlabel('Sample Index')
            plt.ylabel('Expert')
            plt.tight_layout()
            plt.savefig(self.save_dir / 'expert_weights_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Error distribution
        moe_errors = torch.norm(moe_predictions - targets, dim=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(moe_errors.numpy(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('L2 Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of MoE Model Errors')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = moe_errors.mean().item()
        median_error = moe_errors.median().item()
        plt.axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.4f}')
        plt.axvline(median_error, color='orange', linestyle='--', label=f'Median: {median_error:.4f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {self.save_dir}")
    
    def save_results(self, metrics, moe_predictions, targets, expert_weights):
        """Save evaluation results"""
        # Save metrics as JSON
        with open(self.save_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions and targets
        results_data = {
            'moe_predictions': moe_predictions.numpy(),
            'targets': targets.numpy(),
            'expert_weights': expert_weights.numpy()
        }
        
        torch.save(results_data, self.save_dir / 'evaluation_results.pt')
        
        # Save as CSV for easy analysis
        df = pd.DataFrame({
            'steering_pred': moe_predictions[:, 0].numpy(),
            'throttle_pred': moe_predictions[:, 1].numpy(),
            'steering_gt': targets[:, 0].numpy(),
            'throttle_gt': targets[:, 1].numpy(),
            'steering_error': (moe_predictions[:, 0] - targets[:, 0]).numpy(),
            'throttle_error': (moe_predictions[:, 1] - targets[:, 1]).numpy(),
        })
        
        # Add expert weights
        for i, expert_name in enumerate(self.moe_model.gating_network.expert_names):
            df[f'{expert_name}_weight'] = expert_weights[:, i].numpy()
        
        df.to_csv(self.save_dir / 'evaluation_results.csv', index=False)
        
        print(f"Results saved to {self.save_dir}")
    
    def collate_batch(self, batch_list):
        """Collate batch samples"""
        if not batch_list:
            return {}
        
        keys = batch_list[0].keys()
        collated = {}
        
        for key in keys:
            values = [sample[key] for sample in batch_list]
            if isinstance(values[0], torch.Tensor):
                try:
                    collated[key] = torch.stack(values)
                except:
                    collated[key] = values
            else:
                collated[key] = values
        
        return collated
    
    def move_to_device(self, batch):
        """Move batch to device"""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch


def main():
    parser = argparse.ArgumentParser(description='Evaluate MoE model')
    parser.add_argument('--moe_checkpoint', type=str, required=True,
                       help='Path to trained MoE model checkpoint')
    parser.add_argument('--expert_checkpoints', type=str, required=True,
                       help='JSON file with expert checkpoint paths')
    parser.add_argument('--save_dir', type=str, default='eval_results', 
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load expert checkpoint paths
    with open(args.expert_checkpoints, 'r') as f:
        expert_checkpoint_paths = json.load(f)
    
    # Create expert models
    experts = {
        'bdd_detection': BDDDetectionExpert(num_classes=10),
        'bdd_segmentation': BDDSegmentationExpert(),
        'bdd_drivable': BDDDrivableExpert(),
        'nuscenes': NuScenesExpert()
    }
    
    # Create gating network
    gating_network = GatingNetwork(
        num_experts=len(experts),
        gating_type='soft',  # Load from checkpoint if needed
        use_weather_context=True,
        use_traffic_context=True
    )
    
    # Create MoE model
    moe_model = MoEModel(
        experts=experts,
        gating_network=gating_network,
        output_dim=2,
        freeze_experts=True
    )
    
    # Setup validation loader
    val_loader = dataloaders.carla_val_loader
    
    # Create evaluator
    evaluator = MoEEvaluator(
        moe_model=moe_model,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir
    )
    
    # Load checkpoints
    evaluator.load_expert_checkpoints(expert_checkpoint_paths)
    evaluator.load_model_checkpoint(args.moe_checkpoint)
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    print(f"\\nEvaluation completed! Results saved to {save_dir}")


if __name__ == '__main__':
    main()
