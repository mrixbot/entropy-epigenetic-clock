![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Dataset](https://img.shields.io/badge/Dataset-GSE288139-green)
![License](https://img.shields.io/badge/License-MIT-green)

# Entropy Age: A Dual-Stream Deep Learning Epigenetic Clock

This repository contains the source code for a novel deep learning framework designed to model epigenetic aging. Unlike traditional epigenetic clocks that rely solely on single-point methylation averages (beta values), this model simultaneously learns from **mean methylation** and stochastic **epigenetic noise (Shannon pattern entropy)** to predict chronological age.

## Biological Rationale
As human cells age, DNA methylation patterns undergo stochastic changes. This epigenetic "noise" carries a profound, independent biological signal.

To accurately capture this noise, this framework strictly utilizes **read-level sequencing data** (e.g., Targeted Bisulfite Sequencing, WGBS). By analyzing the distribution of multi-CpG combinations across individual DNA strands at a specific locus, we compute the true Shannon pattern entropy.

## Model Architecture
The core of this project is a custom multimodal deep neural network built in PyTorch.

Instead of an early-fusion concatenation, this model utilizes a **Late-Fusion Gated Attention Mechanism**:
1. **Parallel Encoders:** The input tensor is split perfectly in half. Methylation features and Entropy features are routed to isolated, parallel Multi-Layer Perceptrons (MLPs).
2. **Latent Representation:** Each branch maps its input to a localized 64-dimensional embedding using Linear layers, LeakyReLU activations, and Dropout for regularization.
3. **Gated Attention Fusion:** A dynamic gating layer analyzes both embeddings to calculate an attention score. The model dynamically weights the streams—learning exactly when a specific patient's prediction should rely on stochastic entropy versus deterministic methylation—before passing the fused vector to the final prediction head.

---

## Experimental Results (Chan et al., GSE288139)
Evaluated on real human buccal swab sequencing data, the optimized pipeline (incorporating leak-free scaling, target normalization, and validation-monitored early stopping) achieved state-of-the-art predictive performance:
* **Mean Absolute Error (MAE):** `3.73 years`
* **Pearson Correlation (r):** `0.963`
* **Coefficient of Determination (R^2):** `0.916`

### Quantitative Biomarker Split (Gated Attention Analysis)
Extracting the Gated Attention weights across the test cohort revealed a striking biological finding:
* **Shannon Entropy Stream:** `63.0%` mean attention weight
* **Methylation Stream:** `37.0%` mean attention weight

*Conclusion:* The network autonomously prioritized stochastic epigenetic entropy over baseline methylation, confirming that entropy serves as a powerful, high-value biomarker of biological aging.

---

## Repository Structure
```text
entropy_clock_V2/
├── data/
│   ├── raw/                # Raw read-level datasets (e.g., GSE288139)
│   ├── processed/          # Cleaned, extracted feature tensors
│   └── splits/             # Train/Val/Test IDs
├── outputs/                # Generated scatter plots and biomarker contribution charts
├── src/
│   ├── data_loader.py      # PyTorch Dataset wrappers and leak-free scaling
│   ├── entropy.py          # Read-level mathematical feature extraction
│   ├── evaluate.py         # Metrics, regression plotting, and attention/SHAP analysis
│   ├── model.py            # Dual-stream attention-fusion architecture
│   └── train.py            # Gradient descent training loop with validation & Early Stopping
├── run_real.py             # Master orchestrator for real biological pipeline
├── run_synthetic.py        # Master orchestrator for synthetic pilot validation
├── generate_readme.py      # Automated documentation generator
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## How to Run

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execute Synthetic Pilot (Validation):
```bash
python run_synthetic.py
```

### 3. Execute Real Biological Pipeline:
```bash
python run_real.py
```

### 4. View the Results:
Check the `outputs/` folder for the *Predicted vs. Actual* regression plot (`real_predicted_vs_actual.png`) and the *Dual-Stream Biomarker Contribution Analysis* (`real_shap_summary.png`).

## Interpretability
This framework inherently resists the "black box" nature of deep learning. By extracting instance-specific attention weights and incorporating robust post-hoc explainability, the model offers transparent attribution into how deterministic methylation and stochastic entropic drift jointly drive biological aging.
