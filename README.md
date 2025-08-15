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

* **✅ Completed**: 
  - Data collection and preprocessing pipelines for BDD100K, nuScenes, and CARLA have been established. (Waymo and Cosmos datasets are not yet included due to technical issues.)
  - All expert models have been trained and evaluated on their primary datasets (Stage 2), establishing strong baselines.

* **▶️ In Progress**: 
  - Fine-tuning the pre-trained expert models on CARLA data (Stage 3) to adapt them to the CARLA simulator environment.

-----

## 🗺️ Project Roadmap

This project follows a structured, multi-stage development plan.

- ✅ **Stage 1: Data Collection & Preprocessing**
  - Collect and process all primary datasets (BDD100K, nuScenes, etc.).
- ✅ **Stage 2: Expert Training & Evaluation**
  - Train and evaluate the expert models on their respective primary datasets to create strong, specialized baseline models.
- ▶️ **Stage 3: Fine-Tuning on CARLA Data**
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

> Coming soon