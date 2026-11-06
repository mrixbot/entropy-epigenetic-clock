![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Dataset](https://img.shields.io/badge/Dataset-GSE288139-green)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-research%20prototype-orange)

# A Dual-Stream Deep Learning Epigenetic Clock

This repo has the code for my M.Sc. project on modelling epigenetic aging with deep learning. Most epigenetic clocks only use the average methylation at each site (the beta value). This model uses two inputs at the same time: **mean methylation** and **Shannon entropy** of the methylation patterns, which is a measure of how noisy the methylation is. It predicts chronological age from both.

The goal was not to beat other clocks on accuracy. The goal was a model that can say, for each person, how much of its prediction came from methylation and how much from entropy. Please read [Known Limitations](#known-limitations) before using any of the numbers below.

## Biological Rationale
As cells age, DNA methylation becomes more random. A per-site average can't show this, because two samples can have the same mean methylation at a site and still differ a lot in how consistent their reads are.

To measure that noise you need read-level sequencing data (for example targeted bisulfite sequencing or WGBS), because entropy is calculated from the different multi-CpG patterns found across individual DNA strands at one locus.

**Note:** `src/entropy.py` shows how this calculation works on raw reads, but it is not used in the training pipeline. The entropy values I trained on come straight from the per-locus entropy matrix released with GSE288139.

## Model Architecture
The model is a PyTorch network in `src/model.py`. Instead of joining the two inputs at the start, it uses late fusion with a gated attention layer:

1. **Two encoders:** methylation and entropy each go through their own small network. They have the same shape but do not share weights: `Linear(1085 -> 256) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(256 -> 128) -> ReLU`.
2. **Embeddings:** each encoder gives a 128-dimensional vector.
3. **Gated attention:** the two vectors are joined into one 256-vector and passed through `Linear(256 -> 128) -> Tanh -> Linear(128 -> 2) -> Softmax`. The two outputs are positive and add up to 1, so the fused vector is `z = w_m * m + w_e * e`. This is done for every sample, so we can read the split for each person.
4. **Regression head:** `Linear(128 -> 64) -> BatchNorm -> ReLU -> Dropout(0.2) -> Linear(64 -> 1)`.

Training uses Adam with a fixed learning rate of 0.001, a weighted MSE loss (`1 + 2|z|`, which I added to stop the model from just predicting the average age), and early stopping on validation MAE with patience 15. A `ReduceLROnPlateau` scheduler is created in `train.py` but its `step()` line is commented out, so the learning rate never changes.

---

## Experimental Results (Chan et al., GSE288139)
These numbers come from the saved checkpoint `weights/demo_locked_586_weights.pt`, run through `evaluate_weight.py` on the test split (15 of the 100 people, ages 7.2 to 84.4 years, 1,085 loci):

* **Mean Absolute Error (MAE):** `5.86 years`
* **Pearson correlation (r):** `0.920`
* **R^2:** `0.809`

### Linear Baseline
I also fitted an `ElasticNetCV` on the same split and the same 2,170 features (`run_baseline.py`):

* **MAE:** `3.95 years` · **RMSE:** `4.81` · **r:** `0.970`

The baseline is more accurate than my network on every metric. This project does not claim to be state of the art. What the dual-stream model gives that the elastic net can't is the per-person attention split below, but see the cross-validation section before trusting that split.

### Attention Split (Gated Attention)
Average attention weights over the 15 test people:

* **Entropy stream:** `62.9%`
* **Methylation stream:** `37.1%`

All 15 of 15 test people gave more than half their weight to entropy (range 53.4% to 82.1%, SD 8.1 points), so it is not just a few outliers.

### Stream-Level Permutation
I shuffled the values of one stream across people and ran the model again to see how much the error went up:

| Stream permuted | Single shuffle | 200 repeats (mean ± SD) |
| --- | --- | --- |
| Entropy | `+8.75 yrs` | `+5.79 ± 1.51` |
| Methylation | `+3.75 yrs` | `+3.62 ± 1.12` |

Destroying entropy hurts more than destroying methylation, and that held across repeats. The size is less stable. The `+8.75` saved by the pipeline is from one unseeded shuffle and is above the 97.5th percentile of the 200 repeats, so the real ratio is closer to 1.6x than 2.3x.

This is a result for one trained model on 15 test people. It does not measure how much stochastic noise contributes to aging in general.

---

## Five-Fold Cross-Validation (all 100 people)
The saved checkpoint was picked using the test set, so I also ran 5-fold cross-validation on all 100 people. `run_cv_seeds.py` is an extended version of `run_kfold.py`: it loops over seeds, fits `ElasticNetCV` on the same folds, and has a second variant that does the missingness filter and KNN imputation inside each training fold. Folds are stratified on age quartile (random state 42), each test fold has 20 people, and each network trains on 64 people with 16 held out for early stopping. There are 5 seeds per variant, up to 50 epochs (the `run_kfold.py` default), and each stream is permuted 20 times per fold. These are the numbers used in the report, paper and slides.

| | Repo pipeline (filter + impute before split) | Filter + impute inside folds | Frozen checkpoint |
| --- | --- | --- | --- |
| Dual-stream MAE (yr) | 5.97 (seed means 5.72-6.12) | 6.25 (5.98-7.01) | 5.86 |
| Dual-stream r | 0.902 | 0.889 | 0.920 |
| ElasticNetCV MAE (yr) | 4.92 | 4.96 | 3.95 |
| Mean entropy attention weight | 47.8% (42.3-52.6) | 48.5% (38.5-62.6) | 62.9% |
| People routing >50% to entropy | 43% | 43% | 100% (15/15) |
| Permutation cost, entropy / methylation (yr) | +3.38 / +4.57 | +3.52 / +4.18 | +8.75 / +3.75 (single shuffle) |

**About the epochs.** `run_real.py`, which made the checkpoint, allows up to 100 epochs, while `run_kfold.py` uses 50. Early stopping usually ends training much earlier (median epoch 35, range 22 to 71 over 25 fold trainings), so the 50-epoch cap only mattered once. To check, `kfold_run.py` repeats both variants with up to 100 epochs and 100 permutations per fold. That gave MAE 6.24 and 5.91, entropy weight 49.6% and 43.3%, and methylation permutation was costlier in all ten seed/variant runs. The differences from the table are seed noise and the conclusions do not change.

What the cross-validation shows:
* **Accuracy:** 5.86 years is inside the cross-validated range, so picking the checkpoint on the test set did not make the error look much better. The elastic net wins in every fold.
* **Attention split:** the 62.9% for entropy did not come back. Seed means for the repo pipeline are 42.3% to 52.6% (one seed of the strict variant reached 62.6%). The results fit roughly equal use of both streams, which is what my original hypothesis said.
* **Permutation:** methylation costs more than entropy in all 5 seeds of the repo pipeline and 4 of 5 for the strict variant. This is the opposite of the saved checkpoint.
* The seeds change the initialisation and batch order but not the fold split, so the real spread between partitions is probably bigger than shown.

---

## Known Limitations
These are the main reasons to be careful with the numbers above.

* **The checkpoint was chosen using the test set.** `run_real.py` does not set random seeds (those lines are commented out) and `auto_search.py` retrains until the test MAE is under a threshold. So `5.86` is a good run out of many, not an unbiased estimate. The attention and permutation results come from this same checkpoint.
* **Preprocessing is done before the split.** The missingness filter (<= 20% missing) and the `KNNImputer` in `src/process_real_data.py` use all 100 people before the data is split. The scalers are fitted on the training part only, so the leak is limited to filtering and imputation, but it still affects the baseline too.
* **Predictions are squeezed towards the mean.** Predicted vs actual age has slope 0.679 and intercept +12.63. The younger half is overpredicted by 4.78 years on average and the older half underpredicted by 6.22. The `1 + 2|z|` weighting did not fix this.
* **The locus ranking is almost flat.** The entropy encoder's first-layer scores have a coefficient of variation of 4.2%, and the top 50 loci hold 5.01% of the total score against 4.61% if all were equal. The order within the top group means very little, so `outputs/mapped_entropy_genes.csv` is only exploratory.
* **`outputs/god_seed_metrics.txt` is from an older run.** It shows MAE 3.92 / r 0.975 and an attention split near 50/50, which doesn't match the saved checkpoint. I kept it as a record of an earlier restart; it should not be cited.
* **The attention split did not hold up in cross-validation.** Over 5-fold CV the entropy weight averages about 43-50%, not 62.9%, and the permutation order flips. The checkpoint's routing was probably a lucky draw, so I would not read much into it.
* **Only chronological age was available.** There was no biological age, disease or mortality data. **The model is not validated for clinical, forensic or diagnostic use.**

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
│   └── demo_locked_586_weights.pt   # The saved checkpoint used for all reported results
├── src/
│   ├── data_loader.py      # PyTorch Dataset wrappers and train-only scaling
│   ├── entropy.py          # Read-level entropy calculation (for documentation, not used in the pipeline)
│   ├── evaluate.py         # Metrics, regression plot, attention and permutation analysis
│   ├── fetch_data.py       # Downloads the GEO supplementary files
│   ├── fetch_ages.py       # Gets ages from the series matrix
│   ├── model.py            # Dual-stream attention-fusion model
│   ├── process_real_data.py# Cleaning, missingness filter, KNN imputation, stratified split
│   └── train.py            # Training loop with early stopping
├── run_real.py             # Main script for the real data pipeline
├── run_synthetic.py        # Synthetic test run (currently broken: passes an unexpected shap_path argument)
├── run_baseline.py         # ElasticNetCV baseline on the same split
├── run_kfold.py            # Original 5-fold CV script (unseeded, preprocessing outside the folds)
├── run_cv_seeds.py         # 5-fold CV over several seeds, repo and strict variants (50 epochs, 20 permutations); gives the CV tables in the report and paper
├── kfold_run.py            # Repeat of the CV with 100 epochs and 100 permutations (protocols A and B)
├── evaluate_weight.py      # Reproduces all reported results from the saved checkpoint
├── auto_search.py          # Retrains until test MAE is below a threshold
├── extract_pathways.py     # Ranks loci by size of the entropy encoder's first-layer weights
├── map_genes.py            # Maps the top genomic windows to gene symbols with the Ensembl REST API
├── fetch_annotation.py     # Annotation helper
├── inspect_redunt.py       # Checks the feature matrices for redundancy
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
This loads `weights/demo_locked_586_weights.pt`, rebuilds the same stratified split and runs the model once. It is the only script that reproduces the numbers in this README.

### 3. Fit the linear baseline:
```bash
python run_baseline.py
```

### 4. Retrain from scratch (will not give 5.86):
```bash
python run_real.py
```
The seeds are not set, so every run gives a different result. Set them first if you want to compare runs.

### 5. Cross-validation (all 100 people):
```bash
python run_cv_seeds.py repo 0,1,2,3,4      # main tables: repo pipeline, 50 epochs, 20 permutations per fold
python run_cv_seeds.py strict 0,1,2,3,4    # filter + KNN imputation inside each fold (needs data/raw)
python kfold_run.py A 0                    # repeat with 100 epochs, 100 permutations; A = repo CSVs, B = preprocessing inside folds, last number = seed
```
One `run_cv_seeds.py` seed takes under a minute on a CPU and writes `res_<variant>.json`. For the repeat, run `kfold_run.py` for seeds 0 to 4 with both protocols.

### 6. Locus ranking and gene mapping:
```bash
python extract_pathways.py     # writes outputs/top_entropy_cpgs.csv
python map_genes.py            # writes outputs/mapped_entropy_genes.csv
```

### 7. View the results:
Look in the `outputs/` folder for the predicted vs actual plot (`eval_predicted_vs_actual.png`), the attention summary (`eval_attention_summary.png`) and the permutation chart (`eval_stream_importance.png`).

## Interpretability
The attention weight comes straight out of the forward pass, so there is no need for a separate explanation model. This is the one thing the architecture offers that a penalised linear model does not.

It has limits too. The locus ranking in `extract_pathways.py` just adds up the absolute first-layer weights. That ignores everything after the first layer, including the attention module, and it is never tested against data. The gene symbols it gives describe how the network is set up, not how it responds to the inputs, so I would not build any enrichment claim on them without a background set matched to the targeted panel.
