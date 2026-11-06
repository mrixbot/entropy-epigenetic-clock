import numpy as np
import pandas as pd


def _pack_patterns(reads_locus):
    """Encode each read's binary CpG pattern as one integer so patterns can be
    counted with bincount instead of a Python loop. [1,0,1,1] -> 0b1011 -> 11."""
    n_reads, n_cpgs = reads_locus.shape
    weights = (2 ** np.arange(n_cpgs - 1, -1, -1)).astype(np.int64)
    codes = reads_locus.astype(np.int64) @ weights
    return codes, 2 ** n_cpgs


def calculate_metrics_from_reads(reads_array, patient_ids, locus_ids, min_reads=10):
    """Mean methylation and Shannon pattern entropy per patient per locus.

    reads_array: (patients, loci, reads, cpgs), values 0/1, or -1 for missing.
    Returns (mean_methylation_df, entropy_df), both patients x loci.
    Loci with fewer than min_reads usable reads come back as NaN -- entropy on a
    handful of reads is mostly sampling noise.
    """
    n_patients, n_loci, n_reads, n_cpgs = reads_array.shape
    n_states = 2 ** n_cpgs

    mean_meth = np.full((n_patients, n_loci), np.nan)
    entropy = np.full((n_patients, n_loci), np.nan)

    for j in range(n_loci):
        # Pack the whole locus at once, then reshape back per patient
        codes, _ = _pack_patterns(reads_array[:, j, :, :].reshape(-1, n_cpgs))
        codes = codes.reshape(n_patients, n_reads)
        raw_reads = reads_array[:, j, :, :]

        for i in range(n_patients):
            reads_ij = raw_reads[i]  # (n_reads, n_cpgs)

            # Drop whole reads that have any unmapped position, not just the
            # position itself -- a partial read has no valid pattern
            if (reads_ij == -1).any():
                valid_mask = ~np.any(reads_ij == -1, axis=1)
            else:
                valid_mask = np.ones(n_reads, dtype=bool)

            n_valid = valid_mask.sum()
            if n_valid < min_reads:
                continue

            mean_meth[i, j] = reads_ij[valid_mask].mean()

            # H = -sum(p * log2 p) over the 2^n possible patterns.
            # minlength keeps unobserved patterns in the vector; they're
            # filtered out below because 0*log(0) is undefined.
            pattern_codes = codes[i][valid_mask]
            counts = np.bincount(pattern_codes, minlength=n_states)
            probs = counts / counts.sum()
            probs_nonzero = probs[probs > 0]
            entropy[i, j] = -np.sum(probs_nonzero * np.log2(probs_nonzero))

    mean_methylation_df = pd.DataFrame(mean_meth, index=patient_ids, columns=locus_ids)
    entropy_df = pd.DataFrame(entropy, index=patient_ids, columns=locus_ids)
    return mean_methylation_df, entropy_df
