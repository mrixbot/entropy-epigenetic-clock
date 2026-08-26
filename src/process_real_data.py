import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os

def load_and_clean_matrices(meth_path="data/raw/GSE288139_buccalswab_avgmeth.txt.gz", 
                            entropy_path="data/raw/GSE288139_buccalswab_entropy.txt.gz", 
                            missing_threshold=0.2):
    """
    Loads real GEO matrices, aligns patients, drops low-quality loci, 
    and uses K-Nearest Neighbors to impute the remaining missing values.
    """
    print("[*] Loading real biological matrices (this might take a few seconds)...")
    
    # FIX 1: Removed sep='\t' so it defaults to comma-separated
    meth_df = pd.read_csv(meth_path, index_col=0)
    ent_df = pd.read_csv(entropy_path, index_col=0)
    
    # FIX 2: Clean up patient names by removing the '.bam' extension
    meth_df.columns = meth_df.columns.str.replace('.bam', '', regex=False)
    ent_df.columns = ent_df.columns.str.replace('.bam', '', regex=False)
    
    # GEO matrices usually have Loci as rows and Patients as columns.
    # Neural networks need Patients as rows and Features as columns, so we transpose (.T).
    meth_df = meth_df.T
    ent_df = ent_df.T
    
    # Ensure both matrices have the exact same patients
    common_patients = meth_df.index.intersection(ent_df.index)
    meth_df = meth_df.loc[common_patients]
    ent_df = ent_df.loc[common_patients]
    
    print(f"[*] Initial alignment: {len(common_patients)} patients, ~{meth_df.shape[1]} raw loci.")
    
    # ---------------------------------------------------------
    # STEP 1: STRICT QUALITY CONTROL (FILTER)
    # ---------------------------------------------------------
    # Calculate the minimum number of non-NaN values required to keep a locus
    min_valid_patients = len(common_patients) * (1 - missing_threshold)
    
    # Drop columns (loci) that don't meet the threshold
    meth_df = meth_df.dropna(axis=1, thresh=min_valid_patients)
    ent_df = ent_df.dropna(axis=1, thresh=min_valid_patients)
    
    # Since our Dual-Stream Network splits the input perfectly in half, 
    # we MUST have the exact same loci in both the methylation and entropy streams.
    common_loci = meth_df.columns.intersection(ent_df.columns)
    meth_df = meth_df[common_loci]
    ent_df = ent_df[common_loci]
    
    print(f"[*] Post-filtering: Retained {len(common_loci)} high-quality, matching loci.")
    
    # ---------------------------------------------------------
    # STEP 2: SAFE IMPUTATION (FILL)
    # ---------------------------------------------------------
    print("[*] Imputing remaining missing biological gaps using KNN (k=5)...")
    imputer = KNNImputer(n_neighbors=5)
    
    # Impute and reconstruct the pandas DataFrames
    meth_clean = pd.DataFrame(imputer.fit_transform(meth_df), index=meth_df.index, columns=meth_df.columns)
    ent_clean = pd.DataFrame(imputer.fit_transform(ent_df), index=ent_df.index, columns=ent_df.columns)
    
    # Create the data/processed directory to save our clean tensors
    os.makedirs("data/processed", exist_ok=True)
    meth_clean.to_csv("data/processed/clean_meth.csv")
    ent_clean.to_csv("data/processed/clean_entropy.csv")
    
    print("[+] Processing complete! Clean matrices saved to data/processed/")
    
    return meth_clean, ent_clean

if __name__ == "__main__":
    load_and_clean_matrices()