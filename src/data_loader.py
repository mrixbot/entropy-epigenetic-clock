import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. SYNTHETIC "PROOF OF CONCEPT" PIPELINE
# ==========================================
class SyntheticEpigeneticDataset(Dataset):
    def __init__(self, num_samples=100, num_features=1085):
        # Generate fake methylation (beta values 0 to 1) and entropy
        self.meth_data = torch.rand(num_samples, num_features, dtype=torch.float32)
        self.ent_data = torch.rand(num_samples, num_features, dtype=torch.float32)
        # Generate fake ages (20 to 90 years)
        self.ages = torch.randint(20, 90, (num_samples, 1), dtype=torch.float32)
        
        # Inject an artificial biological signal for the model to learn
        # Make the first 10 loci correlate heavily with age
        signal = (self.ages / 100.0).repeat(1, 10)
        self.meth_data[:, :10] += signal
        self.ent_data[:, :10] += (signal * 0.5)

    def __len__(self):
        return len(self.ages)

    def __getitem__(self, idx):
        return self.meth_data[idx], self.ent_data[idx], self.ages[idx]

def get_synthetic_dataloaders(batch_size=16, test_size=0.2):
    print("[*] Initializing Synthetic Proof-of-Concept DataLoader...")
    full_dataset = SyntheticEpigeneticDataset()
    
    train_size = int((1 - test_size) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    feature_names = [f"synth_cg_{i}" for i in range(1085)]
    return train_loader, test_loader, 1085, feature_names

# ==========================================
# 2. REAL BIOLOGICAL PIPELINE (CHAN ET AL.)
# ==========================================
class RealEpigeneticDataset(Dataset):
    def __init__(self, meth_data, ent_data, ages):
        self.meth_data = torch.tensor(meth_data, dtype=torch.float32)
        self.ent_data = torch.tensor(ent_data, dtype=torch.float32)
        # Reshape scaled ages properly
        self.ages = torch.tensor(ages, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.ages)

    def __getitem__(self, idx):
        return self.meth_data[idx], self.ent_data[idx], self.ages[idx]

def get_real_dataloaders(meth_path="data/processed/clean_meth.csv", 
                         ent_path="data/processed/clean_entropy.csv", 
                         age_path="data/processed/clean_ages.csv", 
                         batch_size=16, test_size=0.2):
    print("[*] Initializing Real Biology DataLoader...")
    
    meth_df = pd.read_csv(meth_path, index_col=0)
    ent_df = pd.read_csv(ent_path, index_col=0)
    ages_df = pd.read_csv(age_path, index_col=0)
    
    common_patients = meth_df.index.intersection(ent_df.index).intersection(ages_df.index)
    meth_df = meth_df.loc[common_patients]
    ent_df = ent_df.loc[common_patients]
    ages_df = ages_df.loc[common_patients]
    
    print(f"[*] Final modeling cohort: {len(common_patients)} patients, {meth_df.shape[1]} loci.")
    
    indices = range(len(common_patients))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
    
    meth_scaler = StandardScaler()
    ent_scaler = StandardScaler()
    age_scaler = StandardScaler() # Added a scaler for the Y-variable!
    
    meth_train = meth_scaler.fit_transform(meth_df.iloc[train_idx])
    ent_train = ent_scaler.fit_transform(ent_df.iloc[train_idx])
    meth_test = meth_scaler.transform(meth_df.iloc[test_idx])
    ent_test = ent_scaler.transform(ent_df.iloc[test_idx])
    
    # Scale train and test ages leak-free
    ages_train = age_scaler.fit_transform(ages_df.iloc[train_idx][['Age']]).flatten()
    ages_test = age_scaler.transform(ages_df.iloc[test_idx][['Age']]).flatten()
    
    train_dataset = RealEpigeneticDataset(meth_train, ent_train, ages_train)
    test_dataset = RealEpigeneticDataset(test_meth_dummy := meth_test, ent_test, ages_test) # Keep clean reference
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # We return the age_scaler so evaluation can convert predictions back to real years!
    return train_loader, test_loader, meth_df.shape[1], meth_df.columns.tolist(), age_scaler