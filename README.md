# Alzheimer's Disease Classification with Explainable AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A state-of-the-art deep learning pipeline designed for the classification of Alzheimer's Disease stages from MRI scans. This project leverages the power of **ResNet50** augmented with **Squeeze-and-Excitation (SE) Attention Mechanisms**, ensuring high-performance feature extraction. 

To provide transparency and trust in medical AI, we integrate advanced **Explainable AI (XAI)** techniques.

---

## 🚀 Key Features

### 🧠 Advanced Model Architecture
-   **ResNet50 Backbone**: Industry-standard residual network for robust image classification.
-   **SE Attention Blocks**: Custom squeeze-and-excitation layers that adaptively recalibrate channel-wise feature responses.

### 🔬 Interpretability & Trust (XAI)
-   **Grad-CAM++**: Generates high-resolution heatmaps to visualize exactly which brain regions the model focuses on.
-   **SHAP (GradientExplainer)**: Game-theoretic feature attribution with **custom noise-filtered overlay** for clear anatomical localization.

### ⚙️ Robust Engineering
-   **K-Fold Cross-Validation**: Default 5-fold CV for reliable performance estimation and ensemble modeling.
-   **Modular Design**: Cleanly separated logic for data, modeling, and analysis.
-   **Hardware Agnostic**: Seamlessly switch between **GPU** acceleration and **CPU** inference.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `main.py` | 🚀 **Entry Point**: Orchestrates training, evaluation, and analysis. |
| `config.py` | ⚙️ **Config**: Global hyperparameters and path settings. |
| `models.py` | 🏗️ **Architecture**: ResNet50 implementation with SE blocks. |
| `data.py` | 💾 **Data**: Efficient generators and K-Fold logic. |
| `analysis.py` | 📊 **Analysis**: SHAP explainability implementation. |
| `visualization.py` | 🎨 **Viz**: Grad-CAM++ and plotting utilities. |
| `utils.py` | 🛠️ **Utils**: Logging, seeding, and hardware setup. |

---

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/rabbia67/ALzhimer.git
    cd ALzhimer
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    > **Dependencies**: `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `opencv-python`, `shap`

---

## 💻 Usage

### 🚀 Standard Run (GPU Recommended)
Execute the main pipeline to train the model and generate insights:
```bash
python main.py
```

### 🖥️ CPU-Only Mode
If you lack a dedicated GPU or encounter CUDA issues:
```bash
python main.py --force-cpu
```

---

## 📊 Results & Outputs

All artifacts are automatically saved in `results_advanced/`:

*   **📈 Metrics**: `metrics/` (JSON reports, Confusion Matrices, ROC Curves)
*   **💾 Models**: `models/` (Saved Keras files for each fold)
*   **🧠 Visualizations**:
    *   `gradcam_plus_plus/`: Attention heatmaps overlay.
    *   `shap_analysis/`: Feature contribution plots.

---

## 🔧 Configuration

Customize your experiment in `config.py`:

```python
BATCH_SIZE = 32       # Optimized for GPU
EPOCHS = 60           # Training duration (20 Frozen + 40 Fine-tune)
N_FOLDS = 5           # Cross-validation splits
FORCE_CPU = False     # Override hardware selection
```

---

## ✍️ Authors

**Rabbia Waheed**  
*Developer & Researcher*

-   **GitHub**: [@rabbia67](https://github.com/rabbia67)
-   **Email**: [rabbiawaheed395@gmail.com](mailto:rabbiawaheed395@gmail.com)

---

*Developed for Advanced Medical Imaging Analysis.*
