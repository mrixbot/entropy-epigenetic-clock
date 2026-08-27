import pandas as pd
import numpy as np
import re
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

    meth_df = pd.read_csv(meth_path, index_col=0)
    ent_df = pd.read_csv(entropy_path, index_col=0)

    # CHANGED: the methylation/entropy matrices carry sequencing run
    # suffixes on some patient IDs (e.g. "_S167_L002" -- a sample-sheet
    # index and lane number from the sequencer) that the age metadata's
    # IDs never had. Only stripping ".bam" left these suffixes in place,
    # so 21 patients silently failed to match age metadata and were
    # dropped from the cohort (100 -> 79) with no warning. This regex
    # strips both the .bam extension AND any trailing _S<digits>_L<digits>
    # run suffix, leaving plain patient IDs that match the age metadata.
    def clean_id(col):
        col = col.replace('.bam', '')
        col = re.sub(r'_S\d+_L\d+$', '', col)
        return col

    meth_df.columns = meth_df.columns.map(clean_id)
    ent_df.columns = ent_df.columns.map(clean_id)

    # GEO matrices usually have Loci as rows and Patients as columns.
    # Neural networks need Patients as rows and Features as columns, so we transpose (.T).
    meth_df = meth_df.T
    ent_df = ent_df.T

    # Sanity check: cleaning shouldn't create duplicate patient IDs (which
    # would silently merge two different people). Fail loudly if it does,
    # rather than let pandas quietly pick one.
    if meth_df.index.duplicated().any() or ent_df.index.duplicated().any():
        dupes_meth = meth_df.index[meth_df.index.duplicated()].tolist()
        dupes_ent = ent_df.index[ent_df.index.duplicated()].tolist()
        raise ValueError(
            f"ID cleaning produced duplicate patient IDs -- this would silently "
            f"merge different people. Methylation dupes: {dupes_meth}, "
            f"Entropy dupes: {dupes_ent}. Investigate before proceeding."
        )

    common_patients = meth_df.index.intersection(ent_df.index)
    meth_df = meth_df.loc[common_patients]
    ent_df = ent_df.loc[common_patients]

    print(f"[*] Initial alignment: {len(common_patients)} patients, ~{meth_df.shape[1]} raw loci.")

    # ---------------------------------------------------------
    # STEP 1: STRICT QUALITY CONTROL (FILTER)
    # ---------------------------------------------------------
    min_valid_patients = len(common_patients) * (1 - missing_threshold)

    meth_df = meth_df.dropna(axis=1, thresh=min_valid_patients)
    ent_df = ent_df.dropna(axis=1, thresh=min_valid_patients)

    common_loci = meth_df.columns.intersection(ent_df.columns)
    meth_df = meth_df[common_loci]
    ent_df = ent_df[common_loci]

    print(f"[*] Post-filtering: Retained {len(common_loci)} high-quality, matching loci.")

    # ---------------------------------------------------------
    # STEP 2: SAFE IMPUTATION (FILL)
    # ---------------------------------------------------------
    print("[*] Imputing remaining missing biological gaps using KNN (k=5)...")
    imputer = KNNImputer(n_neighbors=5)

    meth_clean = pd.DataFrame(imputer.fit_transform(meth_df), index=meth_df.index, columns=meth_df.columns)
    ent_clean = pd.DataFrame(imputer.fit_transform(ent_df), index=ent_df.index, columns=ent_df.columns)

    os.makedirs("data/processed", exist_ok=True)
    meth_clean.to_csv("data/processed/clean_meth.csv")
    ent_clean.to_csv("data/processed/clean_entropy.csv")

    print("[+] Processing complete! Clean matrices saved to data/processed/")

    return meth_clean, ent_clean


if __name__ == "__main__":
    load_and_clean_matrices()
