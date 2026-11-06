![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Dataset](https://img.shields.io/badge/Dataset-GSE288139-green)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-research%20prototype-orange)

# A Dual-Stream Deep Learning Epigenetic Clock

This repository contains the source code for a deep learning framework designed to model epigenetic aging. Unlike traditional epigenetic clocks that rely solely on single-point methylation averages (beta values), this model simultaneously learns from **mean methylation** and stochastic **epigenetic noise (Shannon pattern entropy)** to predict chronological age.

The point of the architecture is not accuracy. It is that the model has to state, for every individual, how much of its prediction rests on each of the two signals. See [Known Limitations](#known-limitations) before quoting any number below.

## Biological Rationale
As human cells age, DNA methylation patterns undergo stochastic changes. This epigenetic "noise" carries a biological signal that a per-site average cannot represent: two samples with identical mean methylation at a locus can differ completely in how consistent their individual reads are.

Capturing that noise requires **read-level sequencing data** (e.g. Targeted Bisulfite Sequencing, WGBS), because the Shannon pattern entropy is computed over the distribution of multi-CpG combinations across individual DNA strands at a locus.

**Note on provenance:** `src/entropy.py` implements this calculation from raw reads, but it was written to document the method and is *not* in the modelling path. The entropy values used for training are taken directly from the per-locus entropy matrix released with GSE288139.

## Model Architecture
The core of this project is a multimodal neural network built in PyTorch (`src/model.py`).

Instead of an early-fusion concatenation, this model uses a **Late-Fusion Gated Attention Mechanism**:
1. **Parallel Encoders:** Methylation features and entropy features are routed to isolated, parallel MLPs with no shared weights. Each is `Linear(1085 -> 256) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(256 -> 128) -> ReLU`.
2. **Latent Representation:** Each branch maps its input to a **128-dimensional** embedding.
3. **Gated Attention Fusion:** The two embeddings are concatenated into a 256-vector and passed through `Linear(256 -> 128) -> Tanh -> Linear(128 -> 2) -> Softmax`. The two outputs are positive and sum to 1, so the fusion is a convex combination `z = w_m * m + w_e * e`. Because the gate is evaluated per sample, the split can be read off for each individual patient.
4. **Regression Head:** `Linear(128 -> 64) -> BatchNorm -> ReLU -> Dropout(0.2) -> Linear(64 -> 1)`.

Training uses Adam at a fixed learning rate of 0.001, a weighted MSE loss (`1 + 2|z|`, intended to stop the regressor collapsing to the cohort mean), and early stopping on validation MAE with patience 15. The `ReduceLROnPlateau` scheduler is constructed but its `step()` call is commented out, so the learning rate never changes.

---

## Experimental Results (Chan et al., GSE288139)
All figures below come from the frozen checkpoint `weights/demo_locked_586_weights.pt`, reproduced via `evaluate_weight.py` on the held-out test partition (15 of 100 individuals, aged 7.2 to 84.4 years, 1,085 loci):

* **Mean Absolute Error (MAE):** `5.86 years`
* **Pearson Correlation (r):** `0.920`
* **Coefficient of Determination (R^2):** `0.809`

### Linear Baseline
An `ElasticNetCV` fitted on the identical partition and the identical 2,170 features (`run_baseline.py`):

* **MAE:** `3.95 years` · **RMSE:** `4.81` · **r:** `0.970`

**The baseline is more accurate than the network on every metric.** This repository does not claim state-of-the-art prediction. What the dual-stream model provides and the elastic net cannot is the per-individual attention split below. Read the cross-validation section before relying on that split.

### Quantitative Biomarker Split (Gated Attention Analysis)
Extracting the gated attention weights across the test cohort:

* **Shannon Entropy Stream:** `62.9%` mean attention weight
* **Methylation Stream:** `37.1%` mean attention weight

The mean is not driven by outliers: **all 15 of 15** test individuals routed more than half their weight to entropy, over a range of 53.4% to 82.1% (SD 8.1 points).

### Stream-Level Permutation
Shuffling each stream's values across individuals and re-running inference:

| Stream permuted | Single shuffle | 200 repeats (mean ± SD) |
| --- | --- | --- |
| Entropy | `+8.75 yrs` | `+5.79 ± 1.51` |
| Methylation | `+3.75 yrs` | `+3.62 ± 1.12` |

The **ordering** is reproducible; the magnitude is not. The single `+8.75` figure saved by the pipeline sits above the 97.5th percentile of the repeat distribution because the shuffle is drawn from an unseeded RNG. The honest ratio is roughly 1.6x, not 2.3x.

*Interpretation:* on this cohort the network prioritised stochastic epigenetic entropy over mean methylation, and an independent sensitivity analysis agrees on the direction. This is a property of one fitted model on 15 test individuals, not an estimate of how much stochastic disorder contributes to aging.

---

## Five-Fold Cross-Validation (all 100 individuals)
`run_kfold.py` was run in an extended form (`run_cv_seeds.py`, which loops over seeds, fits `ElasticNetCV` on the same folds and adds a variant that moves the missingness filter and KNN imputer inside each training fold). Folds are stratified on age quartile (random state 42); each held-out fold has 20 individuals; networks train on 64 individuals with 16 for early stopping. Five seeds per variant, up to 50 epochs (the `run_kfold.py` default), each stream permuted 20 times per fold. These are the numbers used in the report, paper and slides.

| | Repo pipeline (filter + impute before split) | Filter + impute inside folds | Frozen checkpoint |
| --- | --- | --- | --- |
| Dual-stream MAE (yr) | 5.97 (seed means 5.72-6.12) | 6.25 (5.98-7.01) | 5.86 |
| Dual-stream r | 0.902 | 0.889 | 0.920 |
| ElasticNetCV MAE (yr) | 4.92 | 4.96 | 3.95 |
| Mean entropy attention weight | 47.8% (42.3-52.6) | 48.5% (38.5-62.6) | 62.9% |
| Individuals routing >50% to entropy | 43% | 43% | 100% (15/15) |
| Permutation cost, entropy / methylation (yr) | +3.38 / +4.57 | +3.52 / +4.18 | +8.75 / +3.75 (single shuffle) |

**Note on epochs.** `run_real.py` (which produced the frozen checkpoint) allows up to 100 epochs and `run_kfold.py` uses 50. Early stopping (patience 15) ends training at a median of epoch 35 (range 22 to 71 over 25 fold trainings), so the cap bound in only one of them and the choice barely matters. As a robustness check, `kfold_run.py` repeats both variants at up to 100 epochs with 100 permutations per fold: MAE 6.24 and 5.91, entropy weight 49.6% and 43.3%, methylation permutation costlier in all ten seed-variant combinations. The differences from the table above are seed-level noise; the conclusions are the same.

What this means:
* **Accuracy:** the frozen checkpoint's 5.86 years is inside the cross-validated range, so selecting it on test error did not inflate the error much. The elastic net beats the network in every fold.
* **Attention routing:** the 62.9% entropy share did not reproduce. The repo-pipeline seed means run from 42.3% to 52.6% (one strict-variant seed reached 62.6%), and the data are consistent with roughly comparable reliance on the two streams, which is where the original hypothesis started.
* **Permutation:** permuting methylation costs more error than permuting entropy in all five seeds of the repo pipeline and four of five of the strict variant, the reverse of the frozen checkpoint.
* The five seeds vary initialisation and batch order but not the fold assignment, so the spread understates partition-to-partition variation.

---

## Known Limitations
These are the reasons the numbers above are optimistic. They are listed here rather than buried because every downstream result inherits them.

* **The checkpoint was selected on the test set.** `run_real.py` does not seed its RNGs (the seed lines are commented out) and `auto_search.py` reruns training until test MAE falls below a threshold. `5.86` is a favourable draw from a distribution of restarts, not an unbiased estimate of generalisation. The attention weights and permutation results inherit this.
* **Preprocessing precedes the split.** The missingness filter (<= 20% missing) and the `KNNImputer` in `src/process_real_data.py` are applied to all 100 individuals before partitioning. Feature scalers *are* fitted on the training partition alone, so the leakage is confined to filtering and imputation — but it is real, and it affects the baseline too.
* **Predictions are compressed towards the mean.** Regression of predicted on actual age gives slope 0.679, intercept +12.63. The younger half is overpredicted by 4.78 years on average and the older half underpredicted by 6.22. The `1 + 2|z|` loss weighting did not prevent this.
* **The locus ranking is nearly flat.** Entropy-encoder first-layer scores have a coefficient of variation of 4.2%; the top 50 loci hold 5.01% of the summed score against 4.61% under a uniform distribution. Rank ordering within the leading group carries little information, so `outputs/mapped_entropy_genes.csv` is exploratory output and nothing more.
* **`outputs/god_seed_metrics.txt` is from a superseded run.** It records MAE 3.92 / r 0.975 and an attention split near 50/50, which contradicts the frozen checkpoint. It is kept as a record of an earlier restart and should not be cited.
* **The attention split does not survive cross-validation.** See the next section: over five-fold CV the entropy weight averages 43-50%, not 62.9%, and the permutation ordering reverses. The frozen checkpoint's routing should be treated as a favourable draw.
* No measure of biological age, morbidity or mortality was available. **This model is not validated for clinical, forensic or diagnostic use.**

---

## Repository Structure
```text
entropy_clock_V2/
├── data/
│   ├── raw/                # GSE288139 avgmeth, entropy and series matrices (.gz)
│   ├── processed/          # clean_meth.csv, clean_entropy.csv, clean_ages.csv
│   └── splits/             # Train/Val/Test IDs
├── outputs/                # Figures, checkpoints from restarts, ranked loci, mapped genes
├── weights/
│   └── demo_locked_586_weights.pt   # The frozen checkpoint all reported results use
├── src/
│   ├── data_loader.py      # PyTorch Dataset wrappers and train-only scaling
│   ├── entropy.py          # Read-level entropy calculation (documentation only, not in pipeline)
│   ├── evaluate.py         # Metrics, regression plotting, attention and permutation analysis
│   ├── fetch_data.py       # GEO supplementary file retrieval
│   ├── fetch_ages.py       # Age metadata parsing from the series matrix
│   ├── model.py            # Dual-stream attention-fusion architecture
│   ├── process_real_data.py# Cleaning, missingness filter, KNN imputation, stratified split
│   └── train.py            # Training loop with early stopping
├── run_real.py             # Master orchestrator for the real biological pipeline
├── run_synthetic.py        # Synthetic pilot (currently broken: passes an unexpected shap_path arg)
├── run_baseline.py         # ElasticNetCV baseline on the identical split
├── run_kfold.py            # Original stratified 5-fold CV harness (unseeded; preprocessing outside the folds)
├── run_cv_seeds.py         # 5-fold CV over seeds ('repo' and 'strict' variants, 50 epochs, 20 permutations): source of the CV tables in the report, paper and this README
├── kfold_run.py            # Robustness repeat of the CV at 100 epochs and 100 permutations (protocols A and B)
├── docs/                   # Project report, IEEE-format paper (PDF + LaTeX source)
├── evaluate_weight.py      # Reproduces all reported results from the frozen checkpoint
├── auto_search.py          # Reruns training until test MAE clears a threshold
├── extract_pathways.py     # Ranks loci by entropy-encoder first-layer weight magnitude
├── map_genes.py            # Maps ranked genomic windows to symbols via the Ensembl REST API
├── fetch_annotation.py     # Annotation helper
├── inspect_redunt.py       # Redundancy inspection of the feature matrices
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## How to Run

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Reproduce the reported results (recommended):
```bash
python evaluate_weight.py
```
This loads `weights/demo_locked_586_weights.pt`, rebuilds the identical stratified split, and runs inference once. It is the only entry point that reproduces the numbers in this README.

### 3. Fit the linear baseline:
```bash
python run_baseline.py
```

### 4. Retrain from scratch (will not reproduce 5.86):
```bash
python run_real.py
```
The RNGs are unseeded, so each run lands somewhere different. Seed them before drawing any conclusion from a single run.

### 5. Cross-validation (all 100 individuals):
```bash
python run_cv_seeds.py repo 0,1,2,3,4      # headline tables: repo pipeline, 50 epochs, 20 permutations per fold
python run_cv_seeds.py strict 0,1,2,3,4    # filter + KNN imputation inside each fold (needs data/raw)
python kfold_run.py A 0                    # robustness repeat: 100 epochs, 100 permutations; A = repo CSVs, B = in-fold preprocessing, second argument = seed
```
Each `run_cv_seeds.py` seed takes under a minute on a CPU and writes `res_<variant>.json`; run `kfold_run.py` for seeds 0 to 4 and both protocols to repeat the robustness check.

### 6. Locus ranking and gene mapping:
```bash
python extract_pathways.py     # writes outputs/top_entropy_cpgs.csv
python map_genes.py            # writes outputs/mapped_entropy_genes.csv
```

### 7. View the results:
Check the `outputs/` folder for the *Predicted vs. Actual* regression plot (`eval_predicted_vs_actual.png`), the attention summary (`eval_attention_summary.png`) and the stream-level permutation chart (`eval_stream_importance.png`).

## Write-up
The full project report and an IEEE-format paper are in [`docs/`](docs/):

* [`docs/Entropy_Clock_Project_Report.pdf`](docs/Entropy_Clock_Project_Report.pdf): the 40-page project report.
* [`docs/Entropy_Clock_IEEE_Paper.pdf`](docs/Entropy_Clock_IEEE_Paper.pdf): the 5-page journal-format paper (LaTeX source in [`docs/ieee_paper/`](docs/ieee_paper/), built with `IEEEtran`).

## Interpretability
The attention weight is an output of the forward pass, not a post-hoc attribution: it is produced on the way to the prediction and requires no surrogate model to recover. That is the one thing this architecture offers that a penalised linear model does not.

It is also the limit of what it offers. The locus-level attribution in `extract_pathways.py` sums absolute first-layer weights, which ignores every downstream transformation including the attention module itself, and never evaluates the weights against data. The gene symbols it produces describe the network's parameterisation, not a measured response to the inputs, and no enrichment claim should be built on them without a background matched to the targeted panel.
