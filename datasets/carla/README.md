# CARLA Autopilot Dataset

## About

[CARLA](https://carla.org/) is an open-source simulator for autonomous driving research. It provides realistic urban environments, dynamic traffic, and flexible sensor setups for collecting high-quality driving data.

This dataset uses CARLA’s autopilot feature to automatically drive the “ego” vehicle through varied scenarios. For each run, the simulator records synchronized camera images, vehicle state (such as speed and steering), and environment context (like weather and traffic).

CARLA-generated datasets like this are ideal for developing and evaluating algorithms in computer vision, imitation learning, and self-driving research.

---

## Quick Start

### 🔄 Download Images from Hugging Face

The entire dataset—including all folders, subfolders, and images—is stored on the Hugging Face Hub as a [Datasets repository](https://huggingface.co/docs/hub/datasets-repos).  
You can **clone the raw CARLA dataset with the original folder structure fully preserved** using Git and Git LFS:

```bash
git lfs install
git clone https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset datasets/carla/raw
````

This will populate the following structure:

```
datasets/carla/raw/
├── run_001/
├── run_002/
└── run_003/
    ...
```

> **Tip:** Make sure you have [Git LFS](https://git-lfs.github.com/) installed so images and other large files are downloaded properly.

## Setup Guide

This guide will walk you through generating and preprocessing a CARLA autopilot dataset.

> 📝 **Note:** This workflow is tested on Linux (Ubuntu) with CARLA = 0.10.0.


### Step 0: Download CARLA 0.10.0

Download the Ubuntu 22 release from [https://github.com/carla-simulator/carla/releases](https://github.com/carla-simulator/carla/releases).

After downloading it to your local machine, move it to your remote machine.

```bash
rsync -avz /path/to/local/CARLA_0.10.0.tar.gz user@remote_ip:/path/to/carla-simulator/
```
  
 In the remote machine, unzip the CARLA file.

```bash
cd /path/to/carla-simulator
tar -xvzf CARLA_0.10.0.tar.gz -C .
```

Make sure to install necessary dependencies on the machine as well.

```bash
pip install --user pygame numpy
```

In `~/.bashrc` and `requirements.txt`, add the following:

```bash
# ~/.bashrc
export CARLA_ROOT=/path/to/carla-simulator

# requirements.txt
/path/to/carla-simulator/PythonAPI/carla/dist/carla-0.10.0-cp310-cp310-linux_x86_64.whl
```

Finally, install CARLA in the project.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Start the CARLA Simulator

Make the launch script executable and run it. (You can override the storage path by setting `CARLA_DATA_PATH`.)

```bash
bash scripts/start_carla.sh
````

To verify the server is up, run:

```bash
python3 scripts/test_carla.py
```

### Step 2: Collect CARLA Autopilot Data

Data is collected by running the robust autopilot collection script.
You can automate multiple collection runs using `scripts/run_carla_scripts.sh` or customize your own commands.

```bash
bash scripts/run_carla_scripts.sh
```

You can also run `collect_autopilot_data.py` directly for fine-grained control:

```bash
python3 scripts/collect_autopilot_data.py --single-run --continue-from 1 --vehicles 30 --walkers 18
```

### Step 3: Preprocess Data

Transform raw images and logs into `.pt` files for efficient deep learning.

```bash
python3 scripts/preprocess_carla.py
```

This will generate PyTorch-ready `.pt` files, split into train/val sets:

```
datasets/carla/preprocessed/
├── train/
└── val/
```

---

## 🗂 Directory Structure

```
datasets/carla/
├── raw/
│   ├── run_001/
│   │   ├── annots/
│   │   │   └── front/
│   │   ├── images/
│   │   │   ├── front/
│   │   │   ├── front_left/
│   │   │   ├── front_right/
│   │   │   └── rear/
│   │   ├── lidar/
│   │   ├── segmentation/
│   │   │   └── front/
│   │   ├── segmentation_vis/
│   │   │   └── front/
│   │   ├── collisions.json
│   │   ├── config.json
│   │   └── vehicle_log.json
│   ├── run_002/
│   └── ...
├── preprocessed/
│   ├── train/
│   │   ├── run_001/
│   │   │   ├── 000000.pt
│   │   │   ├── 000005.pt
│   │   │   └── ...
│   │   └── ...
│   └── val/
│       └── ...
└── README.md
```

---

## ✨ Notes

* **Front camera only:** Preprocessed dataset focuses on the front camera (matching BDD100K), but raw data includes all four views.
* **MoE/Imitation-ready:** Each `.pt` sample contains the image, vehicle state, control, context, and metadata—no extra processing required for PyTorch.
* **Scene variety:** You can customize traffic, pedestrian density, weather, and camera setup per run.
* **Reproducibility:** All scripts are deterministic and log run configuration.
* **Efficient splits:** Training and validation are split by run (not by frame) to avoid temporal leakage.

---

## 📚 References

* [CARLA Simulator](https://carla.org/)
* [CARLA Data Collection Documentation](https://carla.readthedocs.io/en/latest/)
* [CARLA Python API](https://carla.readthedocs.io/en/latest/python_api/)