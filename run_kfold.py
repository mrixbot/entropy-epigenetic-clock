import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

from src.data_loader import load_raw_real_data, create_scaled_dataloaders
from src.model import DualStreamEntropyClockNet
from src.train import train_model

def run_cross_validation(k=5, epochs=50, inner_val_fraction=0.2):
    print("=" * 50)
    print(f" ENTROPY CLOCK: {k}-FOLD CROSS-VALIDATION PIPELINE (100% LEAK-FREE)")
    print("=" * 50)

    # 1. Load Raw Data
    meth_df, ent_df, ages_df = load_raw_real_data()
    input_dim = len(meth_df.columns)
    
    ages_raw = ages_df['Age'].values
    indices = np.arange(len(meth_df))
    age_bins = np.digitize(ages_raw, bins=np.percentile(ages_raw, [25, 50, 75]))

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    fold_maes, fold_rs, fold_r2s = [], [], []
    fold_meth_weights, fold_ent_weights = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, age_bins)):
        print(f"\n--- [FOLD {fold+1}/{k}] ---")

        # 2. Stratify Inner Train/Val
        inner_train_idx, inner_val_idx = train_test_split(
            train_idx, test_size=inner_val_fraction, random_state=42, stratify=age_bins[train_idx]
        )
        print(f"Inner-train: {len(inner_train_idx)}, Inner-val: {len(inner_val_idx)}, Held-out test: {len(test_idx)}")

        # 3. Create isolated dataloaders for this fold
        fold_train_loader, fold_inner_val_loader, fold_test_loader, age_scaler = create_scaled_dataloaders(
            meth_df, ent_df, ages_df, inner_train_idx, inner_val_idx, test_idx
        )

        model = DualStreamEntropyClockNet(input_dim=input_dim)
        model.age_scaler = age_scaler

        model = train_model(model, fold_train_loader, fold_inner_val_loader, epochs=epochs, patience=15)

        model.eval()
        all_preds, all_targets, meth_w, ent_w = [], [], [], []

        with torch.no_grad():
            for meth, ent, age in fold_test_loader:
                preds = model(meth, ent)
                all_preds.extend(preds.numpy().flatten())
                all_targets.extend(age.numpy().flatten())

                m_out = model.meth_stream(meth)
                e_out = model.ent_stream(ent)
                attn = model.attention(torch.cat([m_out, e_out], dim=1))
                meth_w.extend(attn[:, 0].numpy())
                ent_w.extend(attn[:, 1].numpy())

        all_preds = model.age_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
        all_targets = model.age_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()

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

if __name__ == "__main__":
    run_cross_validation(k=5, epochs=50)