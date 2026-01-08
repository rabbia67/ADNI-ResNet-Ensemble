# ADNI-ResNet-Ensemble: Alzheimer's Disease Classification with Explainable AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A deep learning pipeline designed for the automated classification of Alzheimer's Disease stages from MRI scans. This project leverages an ensemble of **ResNet50** models augmented with **Squeeze-and-Excitation (SE) Attention Mechanisms** to ensure robust high-performance feature extraction.

To bridge the gap between "black-box" AI and clinical trust, this framework integrates advanced **Explainable AI (XAI)** techniques, offering transparent visual insights into the decision-making process.

---

## Key Features

### Advanced Model Architecture
*   **Ensemble Learning**: Utilizes specific models trained on 5-fold cross-validation splits to improve generalization and reduce variance.
*   **ResNet50 Backbone**: Industry-standard residual network transfer learning for robust image classification.
*   **SE Attention Blocks**: Custom squeeze-and-excitation layers that adaptively recalibrate channel-wise feature responses, focusing on the most relevant features.

### Interpretability & Trust (XAI)
*   **Grad-CAM++**: Generates high-resolution heatmaps to visualize exactly which brain regions the model focuses on for each prediction.
*   **SHAP (GradientExplainer)**: Game-theoretic feature attribution with a **custom noise-filtered overlay** for precise anatomical localization of disease markers.

### Robust Engineering
*   **K-Fold Cross-Validation**: Implements Stratified K-Fold (default: 5 folds) for reliable performance estimation.
*   **Memory Efficient**: Optimized data generators to handle large medical datasets with minimal RAM usage.
*   **Hardware Agnostic**: Seamlessly switches between **GPU** acceleration (using CUDA) and **CPU** inference based on availability.

---

## Repository Structure

```
ADNI-ResNet-Ensemble/
├── Combined Dataset/          # Dataset directory (Test/Train info)
├── results_advanced/          # Output directory for all artifacts
│   ├── metrics/               # JSON reports, Confusion Matrices, ROC Curves
│   ├── models/                # Saved Keras models for each fold
│   ├── gradcam_plus_plus/     # Attention heatmaps
│   └── shap_analysis/         # SHAP feature plots
├── main.py                    # Entry point for training and evaluation
├── config.py                  # Global configuration and hyperparameters
├── models.py                  # Model definition (ResNet50 + SE blocks)
├── data.py                    # Data loaders and generators
├── analysis.py                # SHAP explainability implementation
├── visualization.py           # Grad-CAM++ and plotting utilities
├── utils.py                   # Helper functions (logging, seeding)
└── requirements.txt           # Project dependencies
```

---

## Installation

### Prerequisites
*   Windows / Linux / macOS
*   Python 3.8 or higher
*   NVIDIA GPU (Recommended for training) with CUDA 11.2+ installed.

### Setup Instructions

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/rabbia67/ADNI-ResNet-Ensemble.git
    cd ADNI-ResNet-Ensemble
    ```

2.  **Create a Virtual Environment** (Optional but Recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Configuration
Before running the project, open `config.py` and ensure the paths point to your data directory.
**Crucial**: Update `BASE_DIR` to match your local path.

```python
# config.py
BASE_DIR = r"path\to\your\ADNI-ResNet-Ensemble"
DATASET_DIR = os.path.join(BASE_DIR, "Combined Dataset")
```

You can also tune hyperparameters like `BATCH_SIZE`, `EPOCHS`, and `N_FOLDS` in this file.

### 2. Standard Execution (GPU Recommended)
Run the complete pipeline (Training → Evaluation → XAI Generation):

```bash
python main.py
```

### 3. CPU-Only Mode
If you do not have a dedicated GPU or are encountering CUDA errors, force the script to use the CPU:

```bash
python main.py --force-cpu
```

---

## Results & Outputs

Upon completion, all results are automatically saved in the `results_advanced/` directory:

*   **Metrics**: Detailed JSON reports showing Accuracy, Precision, Recall, F1-Score, and Kappa for both individual folds and the final ensemble.
*   **Trained Models**: The best performing weights for each fold are saved as `.keras` files.
*   **Visualizations**:
    *   **Grad-CAM++**: See where the model "looks" in the MRI scan.
    *   **SHAP**: Deep learning feature attribution maps.
    *   **Confusion Matrix**: Visual breakdown of classification performance.

---

## Authors & Acknowledgements

**Rabbia Waheed**
*Developer & Researcher*

*   **GitHub**: [@rabbia67](https://github.com/rabbia67)
*   **Email**: [rabbiawaheed395@gmail.com](mailto:rabbiawaheed395@gmail.com)
