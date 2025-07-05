# Self-Driving Model

A comprehensive multi-expert system for autonomous driving that combines perception, motion planning, and end-to-end driving capabilities using multiple datasets and a gating network for intelligent expert selection.

## 🚗 Overview

This project implements a sophisticated self-driving model architecture that leverages multiple specialized experts trained on different datasets (Waymo, nuScenes, BDD100K, CARLA, Cosmos) and uses a gating network to intelligently select the most appropriate expert for different driving scenarios.

### Key Features

- **Multi-Expert Architecture**: Specialized models for perception, motion planning, and end-to-end driving
- **Intelligent Gating**: Dynamic expert selection based on driving context and conditions
- **Multi-Dataset Support**: Training on diverse datasets for robust performance
- **CARLA Integration**: Real-time simulation and testing capabilities
- **Comprehensive Evaluation**: Extensive metrics and visualization tools

## 📁 Project Structure

```
self-driving-model/
│
├── datasets/                    # Dataset storage and organization
│   ├── waymo/                  # Waymo Open Dataset
│   │   ├── perception/         # Perception-focused data
│   │   ├── motion/            # Motion planning data
│   │   └── e2e/               # End-to-end driving data
│   ├── nuscenes/              # nuScenes dataset
│   ├── bdd100k/               # Berkeley DeepDrive dataset
│   ├── carla_expert/          # CARLA simulation data
│   └── cosmos/                # COSMOS dataset
│
├── dataloaders/               # Data loading and preprocessing
│   ├── waymo_loader.py        # Waymo dataset loader
│   ├── nuscenes_loader.py     # nuScenes dataset loader
│   ├── bdd_loader.py          # BDD100K dataset loader
│   ├── carla_loader.py        # CARLA dataset loader
│   └── cosmos_loader.py       # COSMOS dataset loader
│
├── models/                    # Neural network models
│   ├── experts/               # Specialized expert models
│   │   ├── waymo_perception.py
│   │   ├── waymo_motion.py
│   │   ├── waymo_e2e.py
│   │   ├── nuscenes_expert.py
│   │   ├── bdd_expert.py
│   │   └── carla_expert.py
│   ├── gating/                # Gating network components
│   │   ├── gating_network.py  # Expert selection network
│   │   └── feature_fusion.py  # Feature fusion mechanisms
│   └── shared/                # Shared model components
│       ├── encoders.py        # Feature encoders
│       └── decoders.py        # Output decoders
│
├── training/                  # Training scripts and utilities
│   ├── train_expert.py        # Expert model training
│   ├── train_gating.py        # Gating network training
│   ├── loss_functions.py      # Custom loss functions
│   └── utils.py               # Training utilities
│
├── inference/                 # Inference and deployment
│   ├── run_inference.py       # Model inference pipeline
│   └── carla_agent_wrapper.py # CARLA integration wrapper
│
├── eval/                      # Evaluation and metrics
│   ├── metrics.py             # Performance metrics
│   ├── evaluation_loop.py     # Evaluation pipeline
│   └── visualization_tools.py # Result visualization
│
├── scripts/                   # Utility scripts
│   ├── preprocess_waymo.sh    # Waymo preprocessing
│   ├── preprocess_nuscenes.sh # nuScenes preprocessing
│   ├── collect_carla_data.py  # CARLA data collection
│   └── download_cosmos.sh     # COSMOS download script
│
├── docker/                    # Containerization
│   ├── Dockerfile             # Docker image definition
│   ├── docker-compose.yml     # Multi-service setup
│   └── entrypoint.sh          # Container entry point
│
├── configs/                   # Configuration files
│   ├── expert_training.yaml   # Expert training config
│   ├── gating_config.yaml     # Gating network config
│   └── deployment_config.yaml # Deployment settings
│
├── notebooks/                 # Jupyter notebooks
│   ├── visualize_datasets.ipynb    # Dataset exploration
│   └── debug_gating_behavior.ipynb # Gating analysis
│
├── tests/                     # Unit and integration tests
│   ├── test_expert_inference.py
│   ├── test_gating_decision.py
│   └── test_carla_loop.py
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project info
└── LICENSE                    # MIT License
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Docker (for containerized deployment)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/immanuel-peter/self-driving-model.git
   cd self-driving-model
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install CARLA (for simulation)**
   ```bash
   # Follow CARLA installation guide: https://carla.readthedocs.io/
   # Or use Docker:
   docker pull carlasim/carla:latest
   ```

### Docker Setup

For containerized deployment:

```bash
# Build the Docker image
docker build -t self-driving-model .

# Run with docker-compose
docker compose up -d
```

## 📊 Datasets

### Supported Datasets

1. **Waymo Open Dataset**
   - Perception, motion planning, and end-to-end driving
   - Download: [Waymo Open Dataset](https://waymo.com/open/)

2. **nuScenes**
   - Multi-modal autonomous driving dataset
   - Download: [nuScenes](https://www.nuscenes.org/)

3. **BDD100K**
   - Berkeley DeepDrive dataset
   - Download: [BDD100K](https://bdd-data.berkeley.edu/)

4. **CARLA**
   - Simulation environment for data collection
   - Download: [CARLA](https://carla.org/)

5. **COSMOS**
   - Nvidia Physical AI dataset for autonomous driving
   - Download: [Cosmos](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams)

### Data Preprocessing

Run the preprocessing scripts for each dataset:

```bash
# Waymo preprocessing
bash scripts/preprocess_waymo.sh

# nuScenes preprocessing
bash scripts/preprocess_nuscenes.sh

# CARLA data collection
python scripts/collect_carla_data.py

# COSMOS download
bash scripts/download_cosmos.sh
```

## 🏗️ Architecture

### Multi-Expert System

The system consists of specialized expert models:

- **Perception Experts**: Object detection, semantic segmentation
- **Motion Experts**: Trajectory planning, behavior prediction
- **End-to-End Experts**: Direct sensor-to-control mapping

### Gating Network

The gating network dynamically selects the most appropriate expert based on:
- Current driving scenario
- Environmental conditions
- Sensor data quality
- Historical performance

### Feature Fusion

Combines features from multiple experts and sensors for robust decision-making.

## 🚀 Usage

### Training

1. **Train Expert Models**
   ```bash
   python training/train_expert.py --config configs/expert_training.yaml
   ```

2. **Train Gating Network**
   ```bash
   python training/train_gating.py --config configs/gating_config.yaml
   ```

### Inference

1. **Run Inference**
   ```bash
   python inference/run_inference.py --model_path /path/to/model --input /path/to/data
   ```

2. **CARLA Integration**
   ```bash
   python inference/carla_agent_wrapper.py --carla_host localhost --carla_port 2000
   ```

### Evaluation

```bash
python eval/evaluation_loop.py --model_path /path/to/model --dataset waymo
```

## 📈 Performance Metrics

The system evaluates performance using:

- **Perception Metrics**: mAP, IoU, precision/recall
- **Motion Metrics**: ADE, FDE, collision rate
- **End-to-End Metrics**: Success rate, smoothness, safety
- **Gating Metrics**: Expert selection accuracy, switching frequency

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_expert_inference.py
python -m pytest tests/test_gating_decision.py
python -m pytest tests/test_carla_loop.py
```

## 📝 Configuration

### Expert Training Configuration

Edit `configs/expert_training.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Dataset configurations
- Loss function weights

### Gating Configuration

Edit `configs/gating_config.yaml` to adjust:
- Gating network architecture
- Expert selection criteria
- Feature fusion parameters

## 🔧 Development

### Adding New Experts

1. Create expert model in `models/experts/`
2. Add corresponding data loader in `dataloaders/`
3. Update gating network to include new expert
4. Add training configuration

### Adding New Datasets

1. Create dataset loader in `dataloaders/`
2. Add preprocessing script in `scripts/`
3. Update configuration files
4. Add evaluation metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Waymo for the Open Dataset
- nuScenes team for the multi-modal dataset
- Berkeley DeepDrive for BDD100K
- CARLA team for the simulation environment
- COSMOS contributors for the multi-agent dataset

---

**Note**: This is a project.