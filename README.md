# GigaPath AI - WSI Breast Cancer Lesion Analysis Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-red.svg)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Research Only](https://img.shields.io/badge/status-research__only-yellow.svg)](docs/system_disclaimers.md)
[![WSI Supported](https://img.shields.io/badge/data-WSI%20%2F%20TIFF-purple.svg)]()

> **A Research-Grade, Backend-Only Computational Pathology System for Binary Breast Cancer Classification using Multiple Instance Learning (MIL) and Vision Transformers.**

---

## 📑 Table of Contents

1. [Project Overview](#-1-project-overview)
2. [Key Features](#-2-key-features)
3. [System Architecture](#-3-system-architecture)
4. [Folder Structure](#-4-folder-structure)
5. [Data Pipeline](#-5-data-pipeline)
6. [Model Architecture](#-6-model-architecture)
7. [Explainability & Interpretability](#-7-explainability--interpretability)
8. [Inference Output Format](#-8-inference-output-format)
9. [Testing & Evaluation](#-9-testing--evaluation)
10. [Configuration & Customization](#-10-configuration--customization)
11. [Integration Guide](#-11-integration-guide)
12. [Reproducibility & Ethics](#-12-reproducibility--research-ethics)
13. [Limitations & Future Work](#-13-limitations--future-work)
14. [Installation & Execution](#-14-installation--execution)
15. [Final Summary](#-15-final-summary)

---

## 🎯 1. Project Overview

**GigaPath-AI-WSI-Breast-Cancer-Lesion-Analysis** is a specialized, deep learning-based pipeline designed to detect breast cancer lesions in **Whole Slide Images (WSIs)**. Unlike traditional patch-based CNNs that require expensive pixel-level annotations, this system utilizes **Multiple Instance Learning (MIL)** to learn directly from slide-level clinical diagnoses (Cancer / No Cancer).

### Problem Statement
Digital pathology involves analyzing gigapixel-scale images (often >100,000 x 100,000 pixels). Standard deep learning models cannot process these images directly due to memory constraints. Furthermore, obtaining manual segmentations for every tumor cell is clinically impractical.

### Solution
This system adopts a **Weakly Supervised Learning** approach:
1.  **Tiling**: Breaking the gigapixel WSI into thousands of manageable $256 \times 256$ patches.
2.  **Feature Encoding**: Transforming patches into compact embeddings using a foundation model (GigaPath).
3.  **MIL Aggregation**: Using an attention-based neural network to aggregate patch embeddings into a single slide-level prediction, while simultaneously learning which patches are diagnostic (attention scores).

---

## 🌟 2. Key Features

*   **Patch-Based Processing**: High-performance tiling engine optimized for large .tif/.svs files, capable of handling 40x magnification scans.
*   **Foundation Model Integration**: Utilizes pretrained encoders (ResNet50 / GigaPath) for robust feature extraction without end-to-end retraining.
*   **Gated Attention MIL**: Implements the CLAM (Clustering-constrained Attention Multiple instance learning) architecture variants for superior stability and interpretability.
*   **Top-K Sampling Strategy**: Intelligent instance selection to reduce computational load by focusing on the most "salient" tissue regions.
*   **Dual Explainability**:
    *   **Attention Heatmaps**: Visualizes the model's focus at the tile level.
    *   **Grad-CAM (Planned)**: Provides deeper, pixel-level introspection of feature activation.
*   **Confidence Calibration**: Probability-based confidence scores to assist clinical researchers in assessing model certainty.
*   **Modular Design**: Strictly decoupled phases (Preprocessing $\to$ Extraction $\to$ Inference) enabling easy component swapping.
*   **Handover-Ready**: Fully documented, container-friendly, and API-integration ready.

---

## 🏗️ 3. System Architecture

The system follows a strict linear pipeline designed for reproducibility and auditability.

### High-Level Architecture

```ascii
[ WSI Input (.tif) ]
       │
       ▼
[ Phase 0: Preprocessing ] ───▶ Tissue Segmentation & Tiling
       │
       ▼
[ Phase 1: Feature Extractor ] ───▶ Pretrained Encoder (Freeze)
       │
       ▼
[ Feature Bags (.h5) ] ───▶ (N x 2048) Embeddings
       │
       ▼
[ Phase 2: Top-K Sampling ] ───▶ Salience Filtering
       │
       ▼
[ Phase 3: MIL Network ] ───▶ Gated Attention Mechanism
       │
       ├───▶ [ Slide Prediction (0/1) ]
       │
       └───▶ [ Attention Weights (A_i) ]
                  │
                  ▼
[ Phase 4: Visualization ] ───▶ Heatmaps & Overlays
```

### Detailed Inference Flow

```ascii
User Input
    │
    ├── WSI File (.tif) ─────────────────────────────┐
    │                                                │
    │   ┌────────────────────────────────────────┐   │
    ├──▶│ utils.test_inference.py (Orchestrator) │◀──┘
    │   └────────────────────────────────────────┘
    │           │
    │           ▼
    │   ┌─────────────────┐
    │   │ Tile Extractor  │ (src.preprocessing)
    │   └─────────────────┘
    │           │
    │           ▼
    │   ┌─────────────────┐
    │   │ ResNet50/GigaPath │ (src.feature_extraction)
    │   └─────────────────┘
    │           │ Features (HDF5)
    │           ▼
    │   ┌─────────────────┐
    │   │ AttentionMIL    │ (src.mil)
    │   │ (Eval Mode)     │
    │   └─────────────────┘
    │           │
    │           ├───────────────┐
    │           ▼               ▼
    │    [ JSON Prediction ] [ Attention Scores ]
    │                           │
    │                           ▼
    │                    [ Heatmap Generator ]
    │                           │
    │                           ▼
    └──────────────────────▶ [ Output: PNG Heatmaps ]
```

---

## 📂 4. Folder Structure

The project directory is organized to strictly separate code, data, configuration, and outputs.

```text
GigaPath-AI-WSI/
├── configs/                # Configuration YAMLs (Hyperparameters, Paths)
│   └── config.yaml         # Master configuration
│
├── data/                   # MAIN DATA STORAGE (Gitignored)
│   ├── raw_wsi/            # Input Whole Slide Images
│   ├── tiles/              # Extracted patches (transient or saved)
│   ├── features/           # Extracted embeddings (.h5)
│   ├── features_topk/      # Downsampled feature bags
│   └── labels.csv          # Ground truth labels for training
│
├── src/                    # SOURCE CODE MODULES
│   ├── preprocessing/      # Patching, tissue detection (Otsu)
│   ├── feature_extraction/ # CNN/Transformer encoders
│   ├── sampling/           # Top-K / Random sampling logic
│   ├── mil/                # MIL Model definitions & Training loops
│   ├── explainability/     # Heatmap & Grad-CAM generators
│   └── utils/              # Logging, IO, GPU management
│
├── scripts/                # EXECUTABLE SCRIPTS
│   ├── preprocess.py       # Run tiling
│   ├── extract_features.py # Run encoding
│   ├── train_mil.py        # Train model
│   ├── infer_mil.py        # Batch inference
│   ├── test_inference.py   # Single-sample offline test
│   └── generate_heatmaps.py# Visualization
│
├── checkpoints/            # SAVED MODELS
│   └── best_model.pth      # Production-ready MIL weights
│
├── test_data/              # ISOLATED SANDBOX
│   ├── input/              # Drop user files here
│   └── test_results/       # Sandbox outputs
│
├── results/                # EVALUATION OUTPUTS
│   ├── evaluation/         # Confusion matrices, ROC curves
│   └── logs/               # Execution logs
│
└── tests/                  # UNIT TESTS (PyTest)
```

---

## 🔄 5. Data Pipeline

### Step 1: WSI Loading & Tissue Detection
*   **Library**: `OpenSlide` for reading high-res TIFFs.
*   **Logic**:
    1.  Load WSI at a low-resolution level (thumbnail).
    2.  Convert to HSV color space.
    3.  Apply Otsu's thresholding on the Saturation channel to separate tissue from white background.
    4.  Generate a binary tissue mask.

### Step 2: Patch Extraction
*   **Grid**: Slide is divided into a grid of non-overlapping $256 \times 256$ tiles.
*   **Filtering**: Only tiles with $>50\%$ tissue coverage (based on the mask) are kept.
*   **Output**: Patches are either saved as .png files (Research Mode) or held in memory (Production Mode).

### Step 3: Feature Extraction
*   **Input**: Batch of tissue patches.
*   **Model**: ResNet50 (truncated at penultimate layer) or GigaPath.
*   **Output**: A vector of size $2048$ (ResNet) or $1536$ (GigaPath) per patch.
*   **Storage**: Features for the entire slide are batched and saved to a single `.h5` file to prevent filesystem IO bottlenecks.

### Step 4: Bag Creation & Labeling
*   **Bag**: The collection of all $N$ feature vectors from one slide forms a "bag": $X = \{x_1, x_2, ..., x_N\}$.
*   **Label**: The bag inherits the slide's diagnosis (e.g., $Y=1$ for Cancer).

---

## 🧠 6. Model Architecture

We employ a **Gated Attention Multiple Instance Learning** network.

### Concept: Weak Supervision
In standard supervised learning, every input $x$ has a label $y$. In MIL, we only have a label $Y$ for a bag of inputs $X$.
If $Y=0$ (Normal), then **all** $x_i$ must be Normal.
If $Y=1$ (Cancer), then **at least one** $x_i$ must be Cancer.

### Network Components

1.  **Feature Projector**:
    *   Linear layer reducing dimensions (e.g., $2048 \to 512$).
    *   ReLU activation + Dropout.

2.  **Gated Attention Module (Ilse et al. / CLAM)**:
    Computes an attention score $a_k$ for every patch $k$:
    $$a_k = \frac{\exp\{w^T (\tanh(V h_k^T) \odot \text{sigm}(U h_k^T))\}}{\sum_{j=1}^{N} \exp\{w^T (\tanh(V h_j^T) \odot \text{sigm}(U h_j^T))\}}$$
    *   **Tanh Branch**: Learns features.
    *   **Sigmoid Branch**: Acts as a gate to filter irrelevant features.
    *   This provides stable, learnable attention weights.

3.  **Aggregator**:
    Computes the slide representation $H_{slide}$ by weighted sum:
    $$H_{slide} = \sum_{k=1}^{N} a_k h_k$$

4.  **Classifier**:
    *   Final Linear Layer: $H_{slide} \to 1$ (Logit).
    *   Sigmoid Activation: Logit $\to$ Probability.

### Training Details
*   **Loss Function**: Binary Cross Entropy with Logits (`BCEWithLogitsLoss`).
*   **Optimizer**: Adam or AdamW.
*   **Metric**: AUC-ROC (Area Under Curve) is the primary metric due to class imbalance.

---

## 🔍 7. Explainability & Interpretability

Explainability is mandatory in medical AI to verify that the model is detecting cancer cells, not artifacts (e.g., marker pen ink).

### 1. Attention Heatmaps
*   **Mechanism**: The attention scores $a_k$ (from Phase 3) indicate "importance".
*   **Visualization**:
    1.  Normalize $a_k$ to $[0, 1]$ using percentile clipping (robust to outliers).
    2.  Map values to a color map (e.g., **Turbo** or **Jet**).
    3.  Reconstruct the slide grid using tile coordinates.
    4.  Overlay the heatmap on the WSI thumbnail.
*   **Interpretation**: Red regions indicate high suspicion of malignancy (High Attention). Blue regions are considered normal or irrelevant.

### 2. Top-K Tiles
*   The system extracts and displays the top 10 patches with the highest attention scores.
*   Clinicians can rapidly review these 10 patches to verify the diagnosis, rather than scanning the entire slide.

---

## 📄 8. Inference Output Format

The inference pipeline produces structured outputs designed for programmatic parsing.

### 1. Prediction JSON
```json
{
  "slide_id": "test_slide_001",
  "prediction": "Malignant",
  "probability": 0.945,
  "confidence_level": "High",
  "timestamp": "2026-01-12T...",
  "model_version": "v1.0.2"
}
```

### 2. Confidence Scores
*   Stored in CSV format for batch analysis.
*   Contains raw logits, probability ($0.0-1.0$), and calibrated confidence tiers.

### 3. Visualizations
*   `heatmap_overlay.png`: Global view of disease distribution.
*   `heatmap_raw.png`: Uninterpolated attention grid.
*   `top_k_tiles/`: Directory containing high-resolution crops of ROI.

---

## 🧪 9. Testing & Evaluation

### Offline Test Sandbox (`test_data/`)
To ensure safety, we provide a completely isolated "Sandbox" environment.
*   **Purpose**: Test user-uploaded images without contaminating the main `data/` folder.
*   **Mechanism**: `scripts/test_inference.py` automatically routes inputs here.
*   **Rules**: No training allowed. No evaluation metrics (ground truth assumed unavailable).

### Formal Evaluation (`scripts/evaluate_mil.py`)
*   Runs on the Held-Out Test Split.
*   Generates:
    *   **ROC Curve**: To assess sensitivity/specificity trade-off.
    *   **Confusion Matrix**: To visualize false positives/negatives.
    *   **F1 Score**: Harmonic mean of precision and recall.

---

## ⚙️ 10. Configuration & Customization

The system is controlled by `configs/config.yaml`. No code changes are required for standard tuning.

```yaml
# configs/config.yaml

experiment:
  seed: 42
  deterministic: true

preprocessing:
  tile_size: 256
  magnification: 20x
  tissue_threshold: 0.5  # Ignore tiles with <50% tissue

feature_extraction:
  model: "resnet50"      # Options: resnet50, gigapath
  batch_size: 128

mil:
  hidden_dim: 512
  dropout: 0.25
  learning_rate: 1e-4
  epochs: 50
```

To swap the encoder, simply change `model: "resnet50"` to `gigapath` (assuming weights are downloaded).

---

## 🔌 11. Integration Guide

This system is designed as a **Microservice Backend**. It does not provide a UI but exposes clear inputs/outputs for integration.

### Connecting to a Web App (e.g., React/Next.js)
1.  **Frontend**: User uploads a file.
2.  **Backend API**: Fast backend (FastAPI/Flask) receives the file.
3.  **Trigger**: API calls `scripts/test_inference.py --input <filepath>`.
4.  **Polling**: API watches `test_data/test_results/predictions/` for the JSON result.
5.  **Display**: Frontend renders the JSON data and serves the generated PNG heatmaps.

### Connecting to a Desktop App (e.g., Electron/PyQt)
1.  App bundles the Python environment or Docker container.
2.  App invokes the Python scripts via `subprocess` calls.
3.  App reads the JSON/CSV outputs from the local filesystem to display results.

---

## 🔬 12. Reproducibility & Research Ethics

### Determinism
*   All random seeds (Python, NumPy, PyTorch) are fixed in `config.yaml`.
*   Hardware-specific non-deterministic algorithms are disabled where possible.

### Data Privacy
*   The system operates fully offline.
*   No data is sent to external cloud services.
*   Users are responsible for anonymizing WSIs (removing PHI) before processing.

---

## 🚧 13. Limitations & Future Work

*   **Binary Only**: Currently detects only "Cancer" vs "Non-Cancer". It does not grade the cancer (e.g., Gleason score).
*   **Stain Normalization**: Not currently implemented. Variations in H&E staining may affect performance.
*   **Multi-Modal**: Currently supports only H&E images. Future work could include IHC stains.
*   **3D Reconstruction**: Future versions could aggregate serial sections for 3D tumor mapping.

---

## 🚀 14. Installation & Execution

### Prerequisites
*   OS: Windows 10/11 or Linux (Ubuntu 20.04+)
*   GPU: NVIDIA GPU with >6GB VRAM (Required for efficient inference) but the model is  optimised that it will use less than 1gb vram to run so chill out.
*   Python: 3.10+
*   CUDA: 11.8 or higher

### Installation

```bash
# 1. Clone
git clone https://github.com/YourUsername/GigaPath-AI.git
cd GigaPath-AI

# 2. Env
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Deps
pip install -r requirements.txt
pip install openslide-bin  # (Windows only)
```

### Execution (One-Line Sandbox Test)

```bash
python scripts/test_inference.py \
    --input test_data/input/wsi/sample_01.tif \
    --model checkpoints/best_model.pth
```

---

## 🏁 15. Final Summary

The **GigaPath-AI-WSI-Breast-Cancer-Lesion-Analysis** pipeline represents a robust, transparent, and modular approach to computational pathology. By combining the efficiency of patch-based encodings with the interpretability of Attention MIL, it provides a powerful tool for cancer research. Its strict folder structure, fail-safe inference logic, and comprehensive logging make it ready for immediate handover to engineering teams for deployment integration.

---

*Documentation Generated: 2026-01-12 | Version 2.0.0*
