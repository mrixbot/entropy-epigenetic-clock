import pandas as pd
import numpy as np
import re
from sklearn.impute import KNNImputer
import os


def load_and_clean_matrices(meth_path="data/raw/GSE288139_buccalswab_avgmeth.txt.gz",
                             entropy_path="data/raw/GSE288139_buccalswab_entropy.txt.gz",
                             missing_threshold=0.2):
    """Align the two GEO matrices, drop loci with too much missing data, KNN-impute
    the rest, and write the cleaned CSVs.

    Caveat: the filter and the imputer both run over all 100 patients, i.e. before
    the train/test split happens in run_real.py. Test-set info therefore leaks into
    which loci survive and into the imputed values. Moving both inside the fold is
    the fix, but that means retraining everything.
    """
    print("[*] Loading real biological matrices (this might take a few seconds)...")

    meth_df = pd.read_csv(meth_path, index_col=0)
    ent_df = pd.read_csv(entropy_path, index_col=0)

    # Some columns carry a sequencer run suffix (_S167_L002 = sample-sheet index +
    # lane) that the age metadata never had. Stripping only .bam left those in, so
    # 21 patients silently failed to match and the cohort dropped 100 -> 79.
    def clean_id(col):
        col = col.replace('.bam', '')
        col = re.sub(r'_S\d+_L\d+$', '', col)
        return col

    meth_df.columns = meth_df.columns.map(clean_id)
    ent_df.columns = ent_df.columns.map(clean_id)

    # GEO ships loci as rows; torch wants patients as rows
    meth_df = meth_df.T
    ent_df = ent_df.T

    # If cleaning collapsed two IDs into one we'd be merging two people's data,
    # and pandas would just pick one. Fail loudly instead.
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

    # Drop loci missing in more than missing_threshold of patients. Note this is a
    # completeness filter only -- nothing here selects for variance or age signal.
    min_valid_patients = len(common_patients) * (1 - missing_threshold)

    meth_df = meth_df.dropna(axis=1, thresh=min_valid_patients)
    ent_df = ent_df.dropna(axis=1, thresh=min_valid_patients)

    # Keep only loci that survived in both matrices, or the two streams would
    # have different input_dim
    common_loci = meth_df.columns.intersection(ent_df.columns)
    meth_df = meth_df[common_loci]
    ent_df = ent_df[common_loci]

    print(f"[*] Post-filtering: Retained {len(common_loci)} high-quality, matching loci.")

    # Impute what's left. fit_transform refits per matrix, so neighbours are found
    # within meth or within entropy, never across the two.
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
