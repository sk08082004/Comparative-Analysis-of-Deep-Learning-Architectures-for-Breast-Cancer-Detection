# Comparative Analysis of Deep Learning Architectures for Breast Cancer Detection

## Overview

The **Comparative Analysis of Deep Learning Architectures for Breast Cancer Detection** is a high-performance machine learning system designed to preprocess medical imaging data, train multiple deep learning models, and provide an extensible framework for evaluating CNN architectures using a unified pipeline. The system supports multi-class image classification and offers standardized benchmarking, model management, dataset handling, and evaluation tooling for research and production environments.

This platform enables reproducible experimentation with popular CNN architectures and provides modular components for training, fine-tuning, inference, and performance comparison across multiple metrics.

---

## Project Structure

```
breast-cancer-detection/
│
├── SU30MKI.ipynb                    # Main Jupyter notebook with complete workflow
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
│
├── dataset/                          # Dataset directory
│   ├── train/
│   │   ├── benign/
│   │   ├── malignant/
│   │   └── normal/
│   └── test/
│       ├── benign/
│       ├── malignant/
│       └── normal/
│
├── models/                           # Saved trained models
│   ├── vgg16_model.h5
│   ├── densenet121_model.h5
│   ├── mobilenetv2_model.h5
│   └── inceptionv3_model.h5
│
└── results/                          # Training results and visualizations
    ├── plots/
    │   ├── confusion_matrices/
    │   ├── roc_curves/
    │   └── training_history/
    └── metrics/
        └── comparison_report.csv
```

---

## Architecture

The application follows a modular, extensible ML-pipeline architecture with clearly separated functional layers.

### 1. Data Pipeline Layer

- Dataset ingestion and directory parsing
- Standardized preprocessing workflows
- Deterministic train/test splitting
- Real-time augmentation engine
- Unified interfaces for:
  - Image resizing
  - Normalization
  - Augmentation
  - Class distribution handling
  - Dataset-level metadata extraction

### 2. Model Management Layer

**Supported CNN Architectures:**
- VGG16
- DenseNet121
- MobileNetV2
- InceptionV3

**Features:**
- Pretrained ImageNet weight loading
- Customizable classification heads
- Layer freezing/unfreezing logic
- Configurable fine-tuning
- Model checkpointing, versioning, and exporting

### 3. Training & Evaluation Layer

**Standardized, reproducible training pipeline** with two-phase training:
- Feature Extraction
- Fine-Tuning

**Callback integrations:**
- Early stopping
- Learning rate scheduling
- Best-weights saving

**Evaluation engine providing:**
- Accuracy
- Precision
- Recall
- F1-Score
- Cohen's Kappa
- AUC

**Visualization module for:**
- ROC curves
- Confusion matrices
- Loss curves
- Training diagnostics

### 4. Results & Analytics Layer

**Stores:**
- Metrics
- Performance plots
- Comparative reports
- Model profiling data

**Generates:**
- Per-model summaries
- Global comparison charts
- Statistical testing output

### 5. Deployment Layer

- Inference utilities for batch and single-image prediction
- Export formats:
  - Keras SavedModel
  - HDF5
  - Optional ONNX or TensorRT
- Inference profiling:
  - Per-sample latency
  - Throughput
  - GPU/CPU resource usage

---

## Features

### Data Handling

- Multi-class dataset compatibility
- Automatic directory mapping
- GPU-accelerated augmentation
- Configurable image transformations
- Efficient dataloader with caching and prefetching

### Model Training & Comparison

- Uniform model interface for all architectures
- Standardized training hyperparameters
- Feature extraction & fine-tuning workflows
- Model-agnostic evaluation suite
- Auto-saving best performing weights

### Evaluation & Benchmarking

**Metric generation:**
- Accuracy
- Precision
- Recall
- F1-Score
- Kappa score
- Macro AUC

Additional capabilities:
- Statistical significance testing
- Latency and performance profiling
- Model size and parameter comparison
- Detailed confusion matrices and ROC curves

### Extensibility

- Add new models in a plug-and-play fashion
- Modular architecture for easy expansion
- Configurable hyperparameters via central config file
- Reusable data and evaluation pipelines

---

## Technical Stack

### Core Technologies

- Python 3.10+
- TensorFlow / Keras
- CUDA/cuDNN (optional but recommended)
- NumPy & Pandas
- Matplotlib & Seaborn

### Supporting Libraries

- Scikit-Learn – metrics & statistical tools
- tqdm – training progress bars
- Jupyter – notebooks for experimentation

---

## Setup and Installation

### Prerequisites

Ensure the following are installed:
- Python 3.10 or above
- pip or conda
- NVIDIA GPU + CUDA (optional)
- Minimum 8GB RAM

### Environment Setup

1. **Clone the repository:**
```bash
git clone https://github.com//breast-cancer-analysis
cd breast-cancer-analysis
```

2. **Install all dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare dataset directory structure:**
```
dataset/
├── train/
│   ├── benign/
│   ├── malignant/
│   └── normal/
└── test/
    ├── benign/
    ├── malignant/
    └── normal/
```

4. **Optional environment configuration:**
```bash
export MODEL_SAVE_DIR=./models
export DATASET_PATH=./dataset
export IMAGE_SIZE=224
```

---

## Running the Application

### Train All Models

```bash
python src/train.py --all-models
```

### Train Specific Model

```bash
python src/train.py --model vgg16
python src/train.py --model densenet121
python src/train.py --model mobilenetv2
python src/train.py --model inceptionv3
```

### Evaluate Model

```bash
python src/evaluate.py --model inceptionv3
```

### Generate Comparison Report

```bash
python src/generate_report.py
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PREPARATION                             │
│  ┌─────────────┐      ┌──────────────┐      ┌────────────────┐ │
│  │   Dataset   │ -->  │ Augmentation │ -->  │ Preprocessing  │ │
│  │   Loading   │      │   Pipeline   │      │  & Splitting   │ │
│  └─────────────┘      └──────────────┘      └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING (4 Models)                     │
│  ┌─────────┐  ┌──────────────┐  ┌────────────┐  ┌────────────┐│
│  │  VGG16  │  │ DenseNet121  │  │ MobileNetV2│  │InceptionV3 ││
│  └─────────┘  └──────────────┘  └────────────┘  └────────────┘│
│       ↓              ↓                 ↓               ↓        │
│  Feature Extraction → Fine-Tuning → Model Saving                │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                       EVALUATION                                 │
│  ┌────────────┐  ┌─────────────┐  ┌──────────┐  ┌───────────┐ │
│  │ Confusion  │  │  ROC Curve  │  │  Metrics │  │   Kappa   │ │
│  │   Matrix   │  │   & AUC     │  │ Compute  │  │   Score   │ │
│  └────────────┘  └─────────────┘  └──────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL COMPARISON                              │
│        Performance Ranking → Best Model Selection               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Options

*(Training, model, and hardware flags are configurable via the central config file)*

---

## Troubleshooting

**Common Issues:**
- GPU issues
- Out of Memory (OOM) errors
- Slow training performance
- Model saving issues

---

## Future Potential

- PACS integration
- Explainability enhancements
- Additional deployment targets
- Extended model architectures

---

## Authors

**Sunny Kumar**, **Abhijeet Dhanotiya**, **Tushan Kumar Sinha**

— Initial development & maintenance.
