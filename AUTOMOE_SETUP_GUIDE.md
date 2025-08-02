# AutoMoE Implementation Guide

This guide walks you through implementing the AutoMoE (Mixture of Experts) architecture for your self-driving model project, following the roadmap you were given.

## Overview

I've implemented a complete AutoMoE system for you with the following components:

### ✅ What's Now Available

1. **Expert Models** (`models/experts/`):
   - `BDDDetectionExpert` - Object detection expert trained on BDD100K
   - `BDDSegmentationExpert` - Segmentation expert trained on BDD100K  
   - `BDDDrivableExpert` - Drivable area expert trained on BDD100K
   - `NuScenesExpert` - Multimodal (camera + LiDAR) expert trained on NuScenes
   - `CarlaExpert` - CARLA-specific expert with auxiliary tasks

2. **Gating Network** (`models/gating_network.py`):
   - Context-aware gating mechanism
   - Weather and traffic context encoding
   - Soft and hard gating modes
   - Complete MoE model combining experts with gating

3. **Training Infrastructure** (`training/`):
   - `train_experts.py` - Train experts on original datasets
   - `finetune_experts_carla.py` - Fine-tune experts on CARLA data
   - `train_gating_network.py` - Train gating mechanism

4. **Evaluation System** (`eval/`):
   - `evaluate_moe.py` - Comprehensive evaluation with metrics and visualizations

5. **Pipeline Automation** (`scripts/`):
   - `run_automoe_pipeline.py` - Automated pipeline runner

## Quick Start

### Option 1: Automated Pipeline (Recommended)

Run the complete pipeline automatically:

```bash
# Run the full AutoMoE pipeline
python scripts/run_automoe_pipeline.py --step all

# Or run individual steps
python scripts/run_automoe_pipeline.py --step 1  # Train experts
python scripts/run_automoe_pipeline.py --step 2  # Fine-tune on CARLA  
python scripts/run_automoe_pipeline.py --step 3  # Train gating network
python scripts/run_automoe_pipeline.py --step 4  # Evaluate
```

### Option 2: Manual Step-by-Step

## Step 1: Train Experts on Original Datasets

Train each expert on its target dataset:

```bash
# BDD Detection Expert
python training/train_experts.py --expert bdd_detection --epochs 50 --lr 1e-3

# BDD Segmentation Expert  
python training/train_experts.py --expert bdd_segmentation --epochs 50 --lr 1e-3

# BDD Drivable Expert
python training/train_experts.py --expert bdd_drivable --epochs 50 --lr 1e-3

# NuScenes Expert
python training/train_experts.py --expert nuscenes --epochs 50 --lr 1e-3

# CARLA Expert (optional, for comparison)
python training/train_experts.py --expert carla --epochs 50 --lr 1e-3
```

**Expected Output**: Trained expert checkpoints in `checkpoints/{expert_name}/best.pth`

## Step 2: Fine-tune Experts on CARLA Data

Adapt each expert to CARLA's domain:

```bash
# Fine-tune each expert on CARLA data
python training/finetune_experts_carla.py \
    --expert bdd_detection \
    --pretrained_path checkpoints/bdd_detection/best.pth \
    --epochs 20 --lr 1e-4

python training/finetune_experts_carla.py \
    --expert bdd_segmentation \
    --pretrained_path checkpoints/bdd_segmentation/best.pth \
    --epochs 20 --lr 1e-4

# ... repeat for other experts
```

**Expected Output**: Fine-tuned checkpoints in `checkpoints/{expert_name}_carla_finetuned/best_carla_finetuned.pth`

## Step 3: Train Gating Network

First, create an expert checkpoint configuration file (`expert_checkpoints.json`):

```json
{
  "bdd_detection": "checkpoints/bdd_detection_carla_finetuned/best_carla_finetuned.pth",
  "bdd_segmentation": "checkpoints/bdd_segmentation_carla_finetuned/best_carla_finetuned.pth", 
  "bdd_drivable": "checkpoints/bdd_drivable_carla_finetuned/best_carla_finetuned.pth",
  "nuscenes": "checkpoints/nuscenes_carla_finetuned/best_carla_finetuned.pth"
}
```

Then train the gating network:

```bash
python training/train_gating_network.py \
    --expert_checkpoints expert_checkpoints.json \
    --epochs 30 --lr 1e-3 \
    --oracle_labeling \
    --gating_type soft
```

**Expected Output**: Gating network checkpoint in `checkpoints/gating/best_gating.pth`

## Step 4: Evaluate MoE Model

```bash
python eval/evaluate_moe.py \
    --moe_checkpoint checkpoints/gating/best_gating.pth \
    --expert_checkpoints expert_checkpoints.json \
    --save_dir results/
```

**Expected Output**: 
- Evaluation metrics in `results/evaluation_metrics.json`
- Visualizations in `results/*.png`  
- Detailed results in `results/evaluation_results.csv`

## Understanding Your Implementation

### Architecture Overview

```
Input Image + Context
        ↓
   Gating Network ──→ Expert Weights [w1, w2, w3, w4]
        ↓
   Expert Models:
   ├── BDD Detection ──→ Control Output 1
   ├── BDD Segmentation ──→ Control Output 2  
   ├── BDD Drivable ──→ Control Output 3
   └── NuScenes ──→ Control Output 4
        ↓
   Weighted Combination ──→ Final Control (steering, throttle)
```

### Gating Strategy

The gating network considers:
- **Visual features** from the input image
- **Weather context** (rain, fog, visibility)
- **Traffic context** (vehicle density, traffic lights)

### Training Strategy

1. **Expert Training**: Each expert learns its domain (detection, segmentation, etc.)
2. **CARLA Fine-tuning**: Experts adapt to simulation visuals  
3. **Oracle Gating**: Gating network learns which expert performs best per sample
4. **Joint Optimization**: Optional end-to-end fine-tuning

## Key Files You Need to Understand

### Core Model Files
- `models/gating_network.py` - Main MoE architecture
- `models/experts/*.py` - Individual expert models
- `dataloaders/*.py` - Data loading for each dataset

### Training Scripts  
- `training/train_experts.py` - Expert training logic
- `training/finetune_experts_carla.py` - CARLA adaptation
- `training/train_gating_network.py` - Gating training with oracle labeling

### Key Classes
- `MoEModel` - Complete mixture of experts model
- `GatingNetwork` - Context-aware expert selection
- `ExpertTrainer` - Training infrastructure for experts

## Expected Performance

After training, you should see:

1. **Expert Specialization**: Different experts activated for different scenarios
2. **Context Awareness**: Gating decisions based on weather/traffic
3. **Improved Robustness**: Better than single-expert baselines
4. **Domain Adaptation**: Fine-tuned experts perform better on CARLA

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in training scripts
2. **Missing Checkpoints**: Ensure expert training completed successfully  
3. **Data Loading Errors**: Verify your CARLA data is preprocessed correctly
4. **Gating Not Learning**: Try different learning rates or oracle labeling

### Debug Commands

```bash
# Check if experts are working
python -c "from models.experts import BDDDetectionExpert; print('✓ Experts loaded')"

# Test dataloader
python -c "import dataloaders; print('✓ Data loaders working')"

# Verify CARLA data
python -c "
loader = dataloaders.carla_train_loader
batch = next(iter(loader))
print(f'Batch keys: {batch[0].keys() if isinstance(batch, list) else batch.keys()}')
"
```

## Next Steps

1. **Run the pipeline** with your CARLA data
2. **Monitor training** with TensorBoard: `tensorboard --logdir runs/`
3. **Analyze results** in the evaluation outputs
4. **Tune hyperparameters** based on your specific data
5. **Deploy for closed-loop testing** in CARLA simulator

## Advanced Usage

### Custom Expert
Add your own expert by following the pattern in `models/experts/carla_expert.py`

### Different Gating Strategies  
Modify `GatingNetwork` to use different context features or architectures

### End-to-End Training
Implement joint training by unfreezing expert parameters in the gating training script

---

You now have a complete AutoMoE implementation! Start with the automated pipeline and customize as needed for your specific use case.
