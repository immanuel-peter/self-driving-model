"""
AutoMoE Pipeline Runner
This script helps you run the complete AutoMoE training pipeline step by step.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import os


class AutoMoEPipeline:
    def __init__(self, config_file=None):
        self.config = self.load_config(config_file)
        self.device = 'cuda' if self.config.get('use_gpu', True) else 'cpu'
        
    def load_config(self, config_file):
        """Load pipeline configuration"""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "use_gpu": True,
                "batch_size": 32,
                "expert_epochs": 50,
                "finetune_epochs": 20,
                "gating_epochs": 30,
                "learning_rates": {
                    "expert_training": 1e-3,
                    "finetuning": 1e-4,
                    "gating": 1e-3
                },
                "experts": ["bdd_detection", "bdd_segmentation", "bdd_drivable", "nuscenes"],
                "checkpoints_dir": "checkpoints",
                "results_dir": "results"
            }
    
    def run_command(self, command, description=""):
        """Run a command and handle errors"""
        print(f"\\n{'='*60}")
        print(f"RUNNING: {description}")
        print(f"COMMAND: {command}")
        print('='*60)
        
        try:
            result = subprocess.run(command, shell=True, check=True, 
                                  capture_output=False, text=True)
            print(f"✓ SUCCESS: {description}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ FAILED: {description}")
            print(f"Error: {e}")
            return False
    
    def step1_train_experts(self):
        """Step 1: Train experts on their original datasets"""
        print("\\n" + "="*80)
        print("STEP 1: TRAINING EXPERTS ON ORIGINAL DATASETS")
        print("="*80)
        
        success_count = 0
        
        for expert in self.config['experts']:
            command = f"""python training/train_experts.py \\
                --expert {expert} \\
                --epochs {self.config['expert_epochs']} \\
                --lr {self.config['learning_rates']['expert_training']} \\
                --batch_size {self.config['batch_size']} \\
                --device {self.device} \\
                --save_dir {self.config['checkpoints_dir']}"""
            
            if self.run_command(command, f"Training {expert} expert"):
                success_count += 1
        
        print(f"\\nStep 1 Summary: {success_count}/{len(self.config['experts'])} experts trained successfully")
        return success_count == len(self.config['experts'])
    
    def step2_finetune_experts(self):
        """Step 2: Fine-tune experts on CARLA data"""
        print("\\n" + "="*80)
        print("STEP 2: FINE-TUNING EXPERTS ON CARLA DATA")
        print("="*80)
        
        success_count = 0
        
        for expert in self.config['experts']:
            pretrained_path = f"{self.config['checkpoints_dir']}/{expert}/best.pth"
            
            if not Path(pretrained_path).exists():
                print(f"Warning: Pretrained checkpoint not found for {expert}: {pretrained_path}")
                print(f"Skipping fine-tuning for {expert}")
                continue
            
            command = f"""python training/finetune_experts_carla.py \\
                --expert {expert} \\
                --pretrained_path {pretrained_path} \\
                --epochs {self.config['finetune_epochs']} \\
                --lr {self.config['learning_rates']['finetuning']} \\
                --batch_size {self.config['batch_size']} \\
                --device {self.device} \\
                --save_dir {self.config['checkpoints_dir']}"""
            
            if self.run_command(command, f"Fine-tuning {expert} expert on CARLA"):
                success_count += 1
        
        print(f"\\nStep 2 Summary: {success_count} experts fine-tuned successfully")
        return success_count > 0
    
    def step3_create_expert_checkpoint_config(self):
        """Create expert checkpoint configuration file for gating training"""
        expert_checkpoints = {}
        
        for expert in self.config['experts']:
            # Prefer fine-tuned checkpoints, fall back to original
            finetuned_path = f"{self.config['checkpoints_dir']}/{expert}_carla_finetuned/best_carla_finetuned.pth"
            original_path = f"{self.config['checkpoints_dir']}/{expert}/best.pth"
            
            if Path(finetuned_path).exists():
                expert_checkpoints[expert] = finetuned_path
                print(f"Using fine-tuned checkpoint for {expert}")
            elif Path(original_path).exists():
                expert_checkpoints[expert] = original_path
                print(f"Using original checkpoint for {expert}")
            else:
                print(f"Warning: No checkpoint found for {expert}")
        
        # Save checkpoint config
        checkpoint_config_path = "expert_checkpoints.json"
        with open(checkpoint_config_path, 'w') as f:
            json.dump(expert_checkpoints, f, indent=2)
        
        print(f"Expert checkpoint configuration saved to {checkpoint_config_path}")
        return checkpoint_config_path, len(expert_checkpoints) > 0
    
    def step3_train_gating_network(self):
        """Step 3: Train gating network"""
        print("\\n" + "="*80)
        print("STEP 3: TRAINING GATING NETWORK")
        print("="*80)
        
        # Create expert checkpoint config
        checkpoint_config_path, has_experts = self.step3_create_expert_checkpoint_config()
        
        if not has_experts:
            print("No expert checkpoints available for gating training!")
            return False
        
        command = f"""python training/train_gating_network.py \\
            --expert_checkpoints {checkpoint_config_path} \\
            --epochs {self.config['gating_epochs']} \\
            --lr {self.config['learning_rates']['gating']} \\
            --batch_size {self.config['batch_size']} \\
            --device {self.device} \\
            --save_dir {self.config['checkpoints_dir']}/gating \\
            --oracle_labeling \\
            --gating_type soft"""
        
        return self.run_command(command, "Training gating network")
    
    def step4_evaluate_model(self):
        """Step 4: Evaluate MoE model"""
        print("\\n" + "="*80)
        print("STEP 4: EVALUATING MOE MODEL")
        print("="*80)
        
        moe_checkpoint = f"{self.config['checkpoints_dir']}/gating/best_gating.pth"
        expert_checkpoints = "expert_checkpoints.json"
        
        if not Path(moe_checkpoint).exists():
            print(f"MoE checkpoint not found: {moe_checkpoint}")
            return False
        
        if not Path(expert_checkpoints).exists():
            print(f"Expert checkpoints config not found: {expert_checkpoints}")
            return False
        
        command = f"""python eval/evaluate_moe.py \\
            --moe_checkpoint {moe_checkpoint} \\
            --expert_checkpoints {expert_checkpoints} \\
            --save_dir {self.config['results_dir']} \\
            --device {self.device} \\
            --batch_size {self.config['batch_size']}"""
        
        return self.run_command(command, "Evaluating MoE model")
    
    def run_full_pipeline(self):
        """Run the complete AutoMoE pipeline"""
        print("\\n" + "="*80)
        print("STARTING AUTOMOE PIPELINE")
        print("="*80)
        
        # Create directories
        Path(self.config['checkpoints_dir']).mkdir(exist_ok=True)
        Path(self.config['results_dir']).mkdir(exist_ok=True)
        
        # Save config
        with open('pipeline_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        steps = [
            ("Expert Training", self.step1_train_experts),
            ("Expert Fine-tuning", self.step2_finetune_experts),
            ("Gating Network Training", self.step3_train_gating_network),
            ("Model Evaluation", self.step4_evaluate_model)
        ]
        
        successful_steps = 0
        
        for step_name, step_func in steps:
            print(f"\\nStarting: {step_name}")
            
            if step_func():
                print(f"✓ Completed: {step_name}")
                successful_steps += 1
            else:
                print(f"✗ Failed: {step_name}")
                print("Pipeline stopped due to failure.")
                break
        
        print("\\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        print(f"Completed steps: {successful_steps}/{len(steps)}")
        
        if successful_steps == len(steps):
            print("🎉 AutoMoE pipeline completed successfully!")
            print(f"Results saved in: {self.config['results_dir']}")
        else:
            print("❌ Pipeline incomplete. Check logs for errors.")
        
        return successful_steps == len(steps)


def main():
    parser = argparse.ArgumentParser(description='Run AutoMoE training pipeline')
    parser.add_argument('--config', type=str, help='Pipeline configuration file')
    parser.add_argument('--step', type=str, choices=['1', '2', '3', '4', 'all'],
                       default='all', help='Which step to run')
    parser.add_argument('--expert', type=str, help='Run specific expert (for step 1 or 2)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = AutoMoEPipeline(args.config)
    
    if args.step == 'all':
        pipeline.run_full_pipeline()
    elif args.step == '1':
        if args.expert:
            # Run single expert
            pipeline.config['experts'] = [args.expert]
        pipeline.step1_train_experts()
    elif args.step == '2':
        if args.expert:
            pipeline.config['experts'] = [args.expert]
        pipeline.step2_finetune_experts()
    elif args.step == '3':
        pipeline.step3_train_gating_network()
    elif args.step == '4':
        pipeline.step4_evaluate_model()


if __name__ == '__main__':
    main()
