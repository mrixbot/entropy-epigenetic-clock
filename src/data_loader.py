import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# --- synthetic pilot ---------------------------------------------------------
# Only used to check the architecture trains at all before touching real data.

class SyntheticEpigeneticDataset(Dataset):
    def __init__(self, num_samples=100, num_features=1085):
        self.meth_data = torch.rand(num_samples, num_features, dtype=torch.float32)
        self.ent_data = torch.rand(num_samples, num_features, dtype=torch.float32)
        self.ages = torch.randint(20, 90, (num_samples, 1), dtype=torch.float32)

        # Plant an age signal in the first 10 features of each stream, weaker in
        # entropy, so a working model should end up leaning on meth here
        signal = (self.ages / 100.0).repeat(1, 10)
        self.meth_data[:, :10] += signal
        self.ent_data[:, :10] += (signal * 0.5)

    def __len__(self):
        return len(self.ages)

    def __getitem__(self, idx):
        return self.meth_data[idx], self.ent_data[idx], self.ages[idx]

def get_synthetic_dataloaders(batch_size=16, test_size=0.2, val_size=0.2):
    print("[*] Initializing Synthetic Proof-of-Concept DataLoader...")
    full_dataset = SyntheticEpigeneticDataset()
    n = len(full_dataset)
    test_n = int(test_size * n)
    val_n = int(val_size * n)
    train_n = n - test_n - val_n

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    feature_names = [f"synth_cg_{i}" for i in range(1085)]
    return train_loader, val_loader, test_loader, 1085, feature_names


# --- real data (GSE288139) ---------------------------------------------------

class RealEpigeneticDataset(Dataset):
    def __init__(self, meth_data, ent_data, ages):
        self.meth_data = torch.tensor(meth_data, dtype=torch.float32)
        self.ent_data = torch.tensor(ent_data, dtype=torch.float32)
        self.ages = torch.tensor(ages, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.ages)

    def __getitem__(self, idx):
        return self.meth_data[idx], self.ent_data[idx], self.ages[idx]

def load_raw_real_data(meth_path="data/processed/clean_meth.csv",
                       ent_path="data/processed/clean_entropy.csv",
                       age_path="data/processed/clean_ages.csv"):
    """Load the three CSVs and align them on patient ID. No splitting or scaling
    here -- that has to happen after the split or the scalers see the test set."""
    print("[*] Loading raw biological matrices...")
    meth_df = pd.read_csv(meth_path, index_col=0)
    ent_df = pd.read_csv(ent_path, index_col=0)
    ages_df = pd.read_csv(age_path, index_col=0)

    # Intersection, not reindex -- a patient missing from any one file is unusable
    common_patients = meth_df.index.intersection(ent_df.index).intersection(ages_df.index)
    meth_df = meth_df.loc[common_patients]
    ent_df = ent_df.loc[common_patients]
    ages_df = ages_df.loc[common_patients]

    print(f"[*] Final modeling cohort: {len(common_patients)} patients, {meth_df.shape[1]} loci.")
    return meth_df, ent_df, ages_df

def create_scaled_dataloaders(meth_df, ent_df, ages_df, train_idx, val_idx, test_idx, batch_size=16):
    """Fit scalers on train only, then apply to val/test."""
    # Separate scaler per stream: meth is a 0-1 fraction, entropy is in bits, so
    # one shared scaler would squash one of them.
    meth_scaler, ent_scaler, age_scaler = StandardScaler(), StandardScaler(), StandardScaler()

    # fit_transform on train...
    meth_train = meth_scaler.fit_transform(meth_df.iloc[train_idx])
    ent_train = ent_scaler.fit_transform(ent_df.iloc[train_idx])
    ages_train = age_scaler.fit_transform(ages_df.iloc[train_idx][['Age']]).flatten()

    # ...transform only on val and test
    meth_val = meth_scaler.transform(meth_df.iloc[val_idx])
    ent_val = ent_scaler.transform(ent_df.iloc[val_idx])
    ages_val = age_scaler.transform(ages_df.iloc[val_idx][['Age']]).flatten()

    meth_test = meth_scaler.transform(meth_df.iloc[test_idx])
    ent_test = ent_scaler.transform(ent_df.iloc[test_idx])
    ages_test = age_scaler.transform(ages_df.iloc[test_idx][['Age']]).flatten()

    train_loader = DataLoader(RealEpigeneticDataset(meth_train, ent_train, ages_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RealEpigeneticDataset(meth_val, ent_val, ages_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(RealEpigeneticDataset(meth_test, ent_test, ages_test), batch_size=batch_size, shuffle=False)

    # age_scaler goes back to the caller -- metrics have to be computed in years
    return train_loader, val_loader, test_loader, age_scaler
