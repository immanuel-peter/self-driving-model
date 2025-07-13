# BDD100K Dataset

## About

Berkeley DeepDrive 100K (BDD100K) is one of the largest and most diverse driving video datasets for autonomous driving research. Created by UC Berkeley, it contains **100,000 high-resolution driving videos** collected across different weather conditions, times of day, and geographic regions in the United States.

**Key Features:**
- **100K videos** (~1.8M images) with diverse driving scenarios
- **Geographic diversity**: US cities, highways, residential areas
- **Weather conditions**: sunny, cloudy, rainy, snowy, foggy, overcast
- **Time variations**: day, night, dawn, dusk
- **Rich annotations** for multiple computer vision tasks:
  - **Object detection**: 10 categories (car, truck, bus, person, bike, etc.)
  - **Drivable area segmentation**: directly drivable vs. alternative drivable areas
  - **Lane detection**: lane markings and boundaries
  - **Instance segmentation**: pixel-level object masks

The dataset is designed for developing robust autonomous driving systems that can handle real-world driving complexity and environmental variations.

---

## Setup Guide

This guide walks you through downloading and preprocessing the BDD100K dataset for training and evaluation.

### 🔄 Quick Start: Download Raw Images from Hugging Face

If you're only interested in the raw images:

```bash
python scripts/download_bdd100k_raw_images_from_hf.py
```

This will populate the following structure:

```
datasets/bdd100k/raw/images/100k/
├── train/
├── val/
└── test/
```

These images are pulled from the [Hugging Face Hub](https://huggingface.co/datasets/immanuelpeter/bdd100k-raw-images) in split format.

### 🧱 Manual Setup: Download Labels and Organize Full Dataset

If you need the full dataset (labels and official annotations):

#### 1. Create Directory Structure

```bash
mkdir -p datasets/bdd100k/{raw,preprocessed}
cd datasets/bdd100k/raw
```

#### 2. Download from ETHZ Mirror

From the [BDD100K Data Index](https://dl.cv.ethz.ch/bdd100k/data/), download:

* Images:

  * `100k_images_train.zip`
  * `100k_images_val.zip`
  * `100k_images_test.zip`

* Labels:

  * `bdd100k_det_20_labels_trainval.zip`
  * `bdd100k_drivable_labels_trainval.zip`
  * `bdd100k_ins_seg_labels_trainval.zip`

#### 3. Unzip All Files

```bash
unzip 100k_images_train.zip -d .
unzip 100k_images_val.zip -d .
unzip 100k_images_test.zip -d .
unzip bdd100k_det_20_labels_trainval.zip -d .
unzip bdd100k_drivable_labels_trainval.zip -d .
unzip bdd100k_ins_seg_labels_trainval.zip -d .
```

#### 4. Restructure Files

Restructure folders to match this format:

```
datasets/bdd100k/raw/
├── images
│   └── 100k
│       ├── train/
│       ├── val/
│       └── test/
└── labels
    ├── detection2020/
    ├── drivable/
    │   ├── train/
    │   └── val/
    └── segmentation/
        ├── train/
        └── val/
```

#### 🛠️ Use These Terminal Commands:

```bash
mkdir -p images/100k && mv images/{train,val,test} images/100k/
mkdir -p labels/detection2020 && mv labels/det_20/* labels/detection2020/

mkdir -p labels/drivable/{train,val}
mv labels/drivable/colormaps/train/* labels/drivable/train/
mv labels/drivable/colormaps/val/* labels/drivable/val/

mkdir -p labels/segmentation/{train,val}
mv labels/ins_seg/colormaps/train/* labels/segmentation/train/
mv labels/ins_seg/colormaps/val/* labels/segmentation/val/
```

### ⚙️ Preprocessing

To convert raw labels into usable training formats, run:

```bash
python scripts/preprocess_bdd100k.py --detection
python scripts/preprocess_bdd100k.py --drivable
python scripts/preprocess_bdd100k.py --segmentation
```

This will populate:

```
datasets/bdd100k/preprocessed/
├── detection/
│   ├── train/
│   └── val/
├── drivable/
│   ├── train/
│   └── val/
└── segmentation/
    ├── train/
    └── val/
```

#### 📁 Final Directory Structure

```
datasets/bdd100k/
├── raw/
│   ├── images/100k/{train,val,test}
│   └── labels/
│       ├── detection2020/
│       ├── drivable/{train,val}
│       └── segmentation/{train,val}
└── preprocessed/
    ├── detection/{train,val}
    ├── drivable/{train,val}
    └── segmentation/{train,val}
```

---

## ✨ Notes

* Raw image download from Hugging Face is ideal for lightweight experiments or cloud-based pipelines.
* Manual setup is needed if you're using labels for supervised learning tasks.
* Preprocessing scripts convert polygon/mask labels into model-ready formats.

---

## 📚 References

* [Official BDD100K Dataset](https://bdd-data.berkeley.edu/)
* [ETHZ Mirror for BDD100K Data](https://dl.cv.ethz.ch/bdd100k/data/)
* [BDD100K Raw Images](https://huggingface.co/datasets/immanuelpeter/bdd100k-raw-images)