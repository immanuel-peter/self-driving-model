# AutoMoE: Mixture of Experts Self-Driving Model

[![Project Status](https://img.shields.io/badge/status-in%20development-yellowgreen.svg)](https://github.com/immanuel-peter/self-driving-model)

This repository contains the code and experiments for building a modular, multi-task self-driving system based on a **Mixture-of-Experts (MoE)** architecture. The goal is to develop a robust model capable of navigating complex environments in the [CARLA simulator](https://carla.org/).

## 🤖 Architectural Vision

The core of this project is a Mixture-of-Experts model. Instead of a single, monolithic network, this architecture uses:

1. **Specialized "Expert" Models**: A collection of smaller, fine-tuned neural networks, each mastering a specific perception task (e.g., object detection, drivable area segmentation).
2. **Gating Network**: A lightweight network that learns to weigh and combine the outputs of the experts based on the current driving context.

This approach is designed to be more modular, interpretable, and efficient than end-to-end models. The final integrated system will be tested and refined in a high-fidelity simulated environment.

-----

## 📊 Current Status

The project is in its early stages, with the foundational data pipelines and initial model training scripts now complete.

* **✅ Completed**: Data collection and preprocessing pipelines for several large-scale autonomous driving datasets, including **BDD100K**, **nuScenes**, and **CARLA**, have been established. (Unfortunately, I was unable to preprocess **Waymo** datasets due to incompatibility of the Waymo Open Dataset package on various Linux machines. I am looking to come back to it, as well as incorporate the Nvidia **Cosmos** Drive Dreams dataset.)
* **▶️ In Progress**: The training scripts for the BDD100K expert models are fully implemented. This includes a high-performance version utilizing **DistributedDataParallel (DDP)** for efficient multi-GPU training. The immediate next step is to execute these scripts to train the initial set of expert models.

-----

## 🗺️ Project Roadmap

This project follows a structured, multi-stage development plan.

- ✅ **Stage 1: Data Collection & Preprocessing**
  - Collect and process all primary datasets (BDD100K, nuScenes, etc.).
- ▶️ **Stage 2: Expert Training & Evaluation**
  - Train and evaluate the expert models on their respective primary datasets to create strong, specialized baseline models.
- **Stage 3: Fine-Tuning on CARLA Data**
  - Fine-tune the pre-trained experts on data collected from the CARLA simulator to adapt them to the target environment.
- **Stage 4: Gating Network Implementation**
  - Design and implement the gating network architecture responsible for combining expert outputs.
- **Stage 5: Gating Network Training**
  - Train the gating network using the outputs from the fine-tuned experts, likely on a validation set or new data from CARLA.
- **Stage 6: Integrated MoE Simulation**
  - Run the fully integrated MoE model (experts + gating network) in the CARLA simulator to evaluate end-to-end driving performance.
- **Stage 7: Joint Fine-Tuning (Optional)**
  - Explore advanced techniques like Reinforcement Learning or Imitation Learning to jointly fine-tune the gating network and the experts within the simulation.

-----

## ⚙️ Setup and Usage

### 1. Prerequisites

- Git
- Python 3.9+
- NVIDIA GPU with CUDA drivers

### 2. Clone Repository

```bash
git clone https://github.com/immanuel-peter/self-driving-model.git
cd self-driving-model
```

### 3. Setup Environment & Dependencies

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 4. Download and Preprocess Datasets

Follow the setup instructions detailed in the README files within the `datasets/` directory.

### 5. Train an Expert Model

To train an expert on a multi-GPU machine, use the DDP training script with `torchrun`.

**Example: Training the Detection Expert on 2 GPUs**

```bash
torchrun --nproc_per_node=2 --standalone \
    training/train_bdd100k_ddp.py \
    --task detection \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 2e-4 \
    --num_workers 8 \
    --run_name "detection_a100_run"
```

-----

## 📂 Repository Structure

```
├── dataloaders/    # PyTorch DataLoader implementations for each dataset.
├── datasets/       # Instructions and scripts for downloading/preprocessing data.
├── models/         # Neural network architectures for experts and other components.
├── notebooks/      # Jupyter notebooks for exploration and analysis.
├── scripts/        # Utility and processing scripts.
└── training/       # Core training and evaluation loops.
```