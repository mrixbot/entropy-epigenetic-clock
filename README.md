Entropy Age: A Dual-Stream Deep Learning Epigenetic Clock
This repository contains the source code for a novel deep learning framework designed to model epigenetic aging. Unlike traditional epigenetic clocks that rely solely on single-point methylation averages (beta values), this model simultaneously learns from mean methylation and stochastic epigenetic noise (Shannon pattern entropy) to predict chronological age.

Biological Rationale
As human cells age, DNA methylation patterns undergo stochastic changes. This epigenetic "noise" carries a profound, independent biological signal.

To accurately capture this noise, this framework strictly utilizes read-level sequencing data (e.g., Targeted Bisulfite Sequencing, WGBS). By analyzing the distribution of multi-CpG combinations across individual DNA strands at a specific locus, we compute the true Shannon pattern entropy.

Model Architecture: 
The core of this project is a custom multimodal deep neural network built in PyTorch.

Instead of an early-fusion concatenation, this model utilizes a Late-Fusion Gated Attention Mechanism:

Parallel Encoders: The input tensor is split perfectly in half. Methylation features and Entropy features are routed to isolated, parallel Multi-Layer Perceptrons (MLPs).

Latent Representation: Each branch maps its input to a localized 64-dimensional embedding using Linear layers, LeakyReLU activations, and Dropout for regularization.

Gated Attention Fusion: A dynamic gating layer analyzes both embeddings to calculate an attention score (z between 0 and 1). The network dynamically weights the streams—learning exactly when a specific patient's prediction should rely on stochastic entropy versus deterministic methylation—before passing the fused vector to the final prediction head.

📁 Repository Structure
Plaintext
entropy-epigenetic-clock/
├── data/
│   ├── raw/                # Raw read-level datasets (e.g., GSE288139)
│   ├── processed/          # Cleaned, extracted feature tensors
│   └── splits/             # Train/Val/Test IDs
├── outputs/                # Generated scatter plots and SHAP summary plots
├── src/
│   ├── data_loader.py      # PyTorch Dataset wrappers and leak-free scaling
│   ├── entropy.py          # Read-level mathematical feature extraction
│   ├── evaluate.py         # Metrics, Plotting, and SHAP GradientExplainer
│   ├── model.py            # Dual-stream attention-fusion architecture
│   └── train.py            # Gradient descent training loop and validation
├── main.py                 # The master orchestrator script
├── requirements.txt        # Python dependencies
└── .gitignore
Current Status & Pipeline
Phase 1: Synthetic Pilot Validation. The architecture, read-level math, and leak-free pipeline have been strictly validated on an advanced synthetic array simulating biological noise.

Upcoming Commits: 
Phase 2: Real Data Integration. Transitioning the pipeline to ingest and process real targeted bisulfite sequencing datasets (e.g., Chan et al. GSE288139, N=100) using k-fold cross-validation.

Phase 3: Pre-Fusion Feature Selection. Implementing sparse selection techniques (e.g., Elastic Net) to prune uninformative loci before fusion.

How to Run
Currently, the pipeline is configured to run an end-to-end synthetic pilot to validate the architecture and interpretability tools.

1. Install dependencies:

Bash
pip install -r requirements.txt
2. Execute the pipeline:

Bash
python main.py
3. View the results:
Check the outputs/ folder for the Predicted vs. Actual regression plot and the SHAP Summary plot.

Interpretability
This framework inherently resists the "black box" nature of deep learning. Using shap.GradientExplainer (which is robust to the non-linear multiplicative operations in our attention gate), the model generates distinct feature importance scores for both methylation and entropy at every genomic locus.