import numpy as np
import pandas as pd


def _pack_patterns(reads_locus):
    """
    Converts a (reads, CpG_sites) binary array into an integer code per read.
    For example: a 4-CpG pattern [1, 0, 1, 1] becomes binary 1011 -> integer 11.
    This enables fast histogram and frequency counting using np.bincount.
    """
    n_reads, n_cpgs = reads_locus.shape
    # Weights for binary-to-decimal conversion (e.g., [8, 4, 2, 1] for 4 CpGs)
    weights = (2 ** np.arange(n_cpgs - 1, -1, -1)).astype(np.int64)
    codes = reads_locus.astype(np.int64) @ weights
    return codes, 2 ** n_cpgs


def calculate_metrics_from_reads(reads_array, patient_ids, locus_ids, min_reads=10):
    """
    Parameters
    ----------
    reads_array : np.ndarray
        Shape (n_patients, n_loci, n_reads, n_cpgs), values in {0, 1} (or -1 for missing).
    patient_ids : list/array
        Identifiers for each sample/patient.
    locus_ids : list/array
        Identifiers for each genomic locus.
    min_reads : int
        Minimum coverage required; below this threshold, the value is set to NaN.

    Returns
    -------
    mean_methylation_df : pd.DataFrame (patients x loci)
    entropy_df          : pd.DataFrame (patients x loci)
    """
    n_patients, n_loci, n_reads, n_cpgs = reads_array.shape
    n_states = 2 ** n_cpgs

    mean_meth = np.full((n_patients, n_loci), np.nan)
    entropy = np.full((n_patients, n_loci), np.nan)

    for j in range(n_loci):
        # Flatten across all reads for this locus to pack patterns efficiently
        codes, _ = _pack_patterns(reads_array[:, j, :, :].reshape(-1, n_cpgs))
        codes = codes.reshape(n_patients, n_reads)
        raw_reads = reads_array[:, j, :, :]

        for i in range(n_patients):
            reads_ij = raw_reads[i]  # (n_reads, n_cpgs)
            
            # Mask out unmapped/missing reads marked with -1
            if (reads_ij == -1).any():
                valid_mask = ~np.any(reads_ij == -1, axis=1)
            else:
                valid_mask = np.ones(n_reads, dtype=bool)

            n_valid = valid_mask.sum()
            if n_valid < min_reads:
                continue  # Insufficient read coverage

            # 1. Mean Methylation: Fraction of 1s among all valid CpG calls
            mean_meth[i, j] = reads_ij[valid_mask].mean()

            # 2. Pattern Shannon Entropy: Distribution across the 2^n CpG combinations
            pattern_codes = codes[i][valid_mask]
            counts = np.bincount(pattern_codes, minlength=n_states)
            probs = counts / counts.sum()
            probs_nonzero = probs[probs > 0]
            entropy[i, j] = -np.sum(probs_nonzero * np.log2(probs_nonzero))

    mean_methylation_df = pd.DataFrame(mean_meth, index=patient_ids, columns=locus_ids)
    entropy_df = pd.DataFrame(entropy, index=patient_ids, columns=locus_ids)
    return mean_methylation_df, entropy_df