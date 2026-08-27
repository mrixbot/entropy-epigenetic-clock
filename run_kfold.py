import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import os

from src.data_loader import get_real_dataloaders, RealEpigeneticDataset
from src.model import DualStreamEntropyClockNet
from src.train import train_model


def run_cross_validation(k=5, epochs=50, inner_val_fraction=0.2):
    """
    CHANGED from the original run_kfold.py: each fold now carves out its own
    internal validation slice from the fold's TRAINING portion only, and
    trains using that -- never the test fold. The test fold is used exactly
    once, purely for final scoring, after training is completely finished.

    Previously, fold_test_loader was passed to train_model() as the
    validation set (used to pick the "best" checkpoint every epoch) AND
    then reused directly as the test set for final metrics -- meaning the
    reported score was influenced by having seen the test fold during
    training. This mirrors the exact leak that was already fixed in
    run_real.py, which just hadn't propagated to this separate script.

    inner_val_fraction: what fraction of each fold's training data to hold
    out internally for checkpoint selection (default 20%).
    """
    print("=" * 50)
    print(f" ENTROPY CLOCK: {k}-FOLD CROSS-VALIDATION PIPELINE")
    print("=" * 50)

    print("[*] Extracting and pooling data from real dataloaders...")
    # NOTE: written to work whether get_real_dataloaders() returns 6 values
    # (no train_ages_raw) or 7 (with it), so this script doesn't depend on
    # data_loader.py being any particular version.
    loader_outputs = get_real_dataloaders()
    train_loader, val_loader, test_loader, input_dim, feature_names, age_scaler = loader_outputs[:6]

    all_meth, all_ent, all_ages = [], [], []
    for loader in [train_loader, val_loader, test_loader]:
        for meth, ent, age in loader:
            all_meth.append(meth)
            all_ent.append(ent)
            all_ages.append(age)

    all_meth = torch.cat(all_meth, dim=0)
    all_ent = torch.cat(all_ent, dim=0)
    all_ages = torch.cat(all_ages, dim=0)

    dataset = RealEpigeneticDataset(all_meth, all_ent, all_ages)
    dataset.age_scaler = age_scaler
    print(f"[*] Total pooled cohort: {len(dataset)} patients, {input_dim} loci.")

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_maes, fold_rs, fold_r2s = [], [], []
    fold_meth_weights, fold_ent_weights = [], []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f"\n--- [FOLD {fold+1}/{k}] ---")

        # NEW: split this fold's training portion further into an inner
        # train/val split. test_idx is set aside completely and not
        # referenced again until final scoring below.
        inner_train_idx, inner_val_idx = train_test_split(
            train_idx, test_size=inner_val_fraction, random_state=42
        )
        print(f"Inner-train: {len(inner_train_idx)}, Inner-val: {len(inner_val_idx)}, "
              f"Held-out test: {len(test_idx)}")

        inner_train_sub = Subset(dataset, inner_train_idx)
        inner_val_sub = Subset(dataset, inner_val_idx)
        test_sub = Subset(dataset, test_idx)

        fold_train_loader = DataLoader(inner_train_sub, batch_size=16, shuffle=True)
        fold_inner_val_loader = DataLoader(inner_val_sub, batch_size=16, shuffle=False)
        fold_test_loader = DataLoader(test_sub, batch_size=16, shuffle=False)

        model = DualStreamEntropyClockNet(input_dim=input_dim)
        model.age_scaler = dataset.age_scaler

        # CHANGED: checkpoint selection now uses fold_inner_val_loader,
        # never fold_test_loader
        model = train_model(model, fold_train_loader, fold_inner_val_loader,
                             epochs=epochs, patience=15)

        # Test fold touched here for the first and only time
        model.eval()
        all_preds, all_targets = [], []
        meth_w, ent_w = [], []

        with torch.no_grad():
            for meth, ent, age in fold_test_loader:
                preds = model(meth, ent)
                all_preds.extend(preds.numpy().flatten())
                all_targets.extend(age.numpy().flatten())

                m_out = model.meth_stream(meth)
                e_out = model.ent_stream(ent)
                combined = torch.cat([m_out, e_out], dim=1)
                attn = model.attention(combined)

                meth_w.extend(attn[:, 0].numpy())
                ent_w.extend(attn[:, 1].numpy())

        all_preds = np.array(all_preds).reshape(-1, 1)
        all_targets = np.array(all_targets).reshape(-1, 1)

        if hasattr(model, 'age_scaler'):
            all_preds = model.age_scaler.inverse_transform(all_preds).flatten()
            all_targets = model.age_scaler.inverse_transform(all_targets).flatten()

        mae = mean_absolute_error(all_targets, all_preds)
        r, _ = pearsonr(all_targets, all_preds) if len(np.unique(all_preds)) > 1 else (0.0, 0)
        r2 = r2_score(all_targets, all_preds)

        fold_maes.append(mae)
        fold_rs.append(r)
        fold_r2s.append(r2)
        fold_meth_weights.append(np.mean(meth_w))
        fold_ent_weights.append(np.mean(ent_w))

        print(f"Fold {fold+1} Results (genuinely held-out) -> MAE: {mae:.2f} yrs | r: {r:.3f} | R2: {r2:.3f}")

    print("\n" + "=" * 50)
    print(f" CROSS-VALIDATION SUMMARY ({k} FOLDS, LEAK-FREE)")
    print("=" * 50)
    print(f" Mean MAE:               {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f} years")
    print(f" Mean Pearson (r):       {np.mean(fold_rs):.3f} ± {np.std(fold_rs):.3f}")
    print(f" Mean R-Squared (R2):    {np.mean(fold_r2s):.3f} ± {np.std(fold_r2s):.3f}")
    print(f" Mean Methylation Weight: {np.mean(fold_meth_weights)*100:.1f}%")
    print(f" Mean Entropy Weight:     {np.mean(fold_ent_weights)*100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    run_cross_validation(k=5, epochs=50)
