# nuScenes Dataset

## About

nuScenes is a large-scale autonomous driving dataset developed by Motional. It contains **1000 scenes** of driving data collected in Boston and Singapore, each lasting **20 seconds**. The dataset provides comprehensive sensor data from **6 cameras**, **1 LiDAR**, **5 radars**, plus **GPS/IMU** for full 360° perception.

**Key Features:**
- **850 scenes** for training/validation, **150 scenes** for testing
- **1.4M camera images** with 3D bounding box annotations
- **390K LiDAR sweeps** with point cloud data
- **23 object classes** (cars, pedestrians, bikes, etc.)
- **Multimodal sensor fusion** capabilities
- **Temporal sequences** for tracking and prediction tasks

The dataset is designed for autonomous driving research including **3D object detection**, **tracking**, **motion prediction**, and **sensor fusion**.

---

## Setup Guide

This guide walks you through downloading, verifying, and preprocessing the nuScenes dataset for training and evaluation with PyTorch.

### 📦 Step 1: (Optional) Download the Mini Split

Use the **Mini split** (~1GB) for quick prototyping or exploration.

```bash
tar -xzf v1.0-mini.tgz -C path/to/nuscenes/mini
```

> 💡 Use the included notebook `notebooks/explore_nuscenes.ipynb` to visualize samples from the Mini split.

### 🧠 Step 2: Download the Trainval 1.0 Split

You only need:

* ✅ The **metadata**
* ✅ 2–3 parts of the full `trainval` set (e.g., Part 1, 3, 6)

Download from [nuScenes Download Page](https://www.nuscenes.org/download), then extract:

```bash
tar -xzf v1.0-trainval_meta.tgz -C path/to/nuscenes/raw
tar -xzf v1.0-trainval01_blobs.tgz -C path/to/nuscenes/raw
tar -xzf v1.0-trainval03_blobs.tgz -C path/to/nuscenes/raw
tar -xzf v1.0-trainval06_blobs.tgz -C path/to/nuscenes/raw
```

### ✅ Step 3: Verify and Preprocess

1. **Check available scenes**

   ```bash
   python scripts/check_nuscenes.py
   ```

   This script will count how many scenes are usable (based on downloaded data).

2. **Preprocess the dataset**

   ```bash
   python scripts/preprocess_nuscenes.py
   ```

   This will convert nuScenes into `.pt` files, organized into:

   ```
   datasets/nuscenes/preprocessed/
   ├── train/    
   └── val/      
   ```

#### 🗂 Directory Structure Overview

```
datasets/nuscenes/
├── mini/                # Optional mini split
├── raw/                 # Raw downloaded .jpg/.bin files
├── preprocessed/        # Cached .pt files for fast training
│   ├── train/
│   └── val/
└── README.md            # This file
```

---

## ✨ Notes

* **Mini split** is ideal for prototyping and development
* **Official train/val splits** (700/150 scenes) are automatically used during preprocessing for consistent evaluation.
* **Preprocessing** converts raw sensor data into PyTorch tensors for 3-5x faster training data loading.
* **Multimodal fusion** is supported with aligned camera images, LiDAR point clouds, and calibration matrices.
* Only **2-3 data parts** are needed instead of all 10 parts, reducing download size significantly.

---

## 📚 References

* [Official nuScenes Dataset](https://www.nuscenes.org/)
* [nuScenes devkit Documentation](https://github.com/nutonomy/nuscenes-devkit)
* [Download Page](https://www.nuscenes.org/download)