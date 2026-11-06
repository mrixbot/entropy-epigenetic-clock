import torch
import pandas as pd
import numpy as np
import os

from src.data_loader import load_raw_real_data
from src.model import DualStreamEntropyClockNet

def extract_top_cpgs(weight_path="outputs/demo_locked_586_weights.pt", top_k=50):
    """Rank loci by how much first-layer weight the entropy stream puts on them.

    Rough proxy only: it reads one layer, ignores everything downstream including
    the attention gate, and never touches the data. The resulting scores also sit
    in a narrow band, so the ordering near the top is not very stable.
    """
    print("==================================================")
    print(" BIOLOGICAL MAPPING: EXTRACTING TOP ENTROPY LOCI")
    print("==================================================")

    # Loaded purely for the column names -- the matrices themselves aren't used
    print("[*] Loading feature names...")
    meth_df, _, _ = load_raw_real_data()
    feature_names = meth_df.columns.tolist()
    input_dim = len(feature_names)

    model = DualStreamEntropyClockNet(input_dim=input_dim)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu'), weights_only=True))


    try:
        entropy_weights = model.ent_stream[0].weight.detach().numpy()  # (256, n_loci)
    except AttributeError:
        print("[!] Note: Please update the script with your exact entropy stream first-layer name.")
        return

    # Sum |w| down the 256 hidden units to get one score per locus
    feature_importance = np.sum(np.abs(entropy_weights), axis=0)

    importance_df = pd.DataFrame({
        'CpG_Locus': feature_names,
        'Importance_Score': feature_importance
    }).sort_values(by='Importance_Score', ascending=False)

    os.makedirs("outputs", exist_ok=True)
    out_file = "outputs/top_entropy_cpgs.csv"
    top_cpgs = importance_df.head(top_k)
    top_cpgs.to_csv(out_file, index=False)

    print(f"\n[+] Successfully saved top {top_k} entropy-driving CpG loci to {out_file}")
    print("\nTop 5 Most Crucial CpG Loci:")
    print(top_cpgs.head(5).to_string(index=False))

if __name__ == "__main__":
    extract_top_cpgs()
