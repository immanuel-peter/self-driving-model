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

This project follows a structured, multi-stage development plan that separates **perception** (seeing) from **control** (driving).

- ✅ **Stage 1: Data Collection & Preprocessing**
  - Collect and process all primary datasets (BDD100K, nuScenes, CARLA raw data).
- ✅ **Stage 2: Expert Training & Evaluation**
  - Train and evaluate the expert models (detection, segmentation, drivable) on their respective primary datasets to create strong, specialized baselines.
- ▶️ **Stage 3: CARLA Expert Adaptation**
  - Generate CARLA supervision (pseudo-labels from BDD experts or GT from CARLA sensors).
  - Fine-tune experts on CARLA to reduce domain gap and produce clean outputs in the simulator environment.
- **Stage 4: Policy Head Development**
  - Train a CARLA-specific control module (BC, IL, or RL) to turn perception outputs into `{steer, throttle, brake}` commands.
- **Stage 5: Gating Network Implementation**
  - Design and implement the gating network architecture responsible for combining expert outputs before the policy head.
- **Stage 6: Gating Network Training**
  - Train the gating network on CARLA-adapted expert outputs to improve expert routing in the target domain.
- **Stage 7: Integrated MoE + Policy Simulation**
  - Wire perception experts, gating network, and control module into CARLA’s synchronous simulation loop.
  - Evaluate closed-loop driving performance (route completion, infractions/km, jerk).
- **Stage 8: Joint Fine-Tuning (Optional)**
  - Use DAgger, RL, or IL to jointly fine-tune perception, gating, and control in-sim for robustness.


-----

## ⚙️ Setup and Usage

> Coming soon