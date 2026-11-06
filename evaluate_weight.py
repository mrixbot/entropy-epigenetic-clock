import random
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

# Seeded here (unlike run_real.py) so the reported figures are reproducible --
# matters because evaluate.py's permutation step draws from the global RNG.
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from src.data_loader import load_raw_real_data, create_scaled_dataloaders
from src.model import DualStreamEntropyClockNet
from src.evaluate import evaluate_and_plot

def main():
    """Score a saved checkpoint without retraining anything."""
    weight_file = "weights/demo_locked_586_weights.pt"

    print("==================================================")
    print(" ENTROPY CLOCK: STANDALONE WEIGHT EVALUATION")
    print("==================================================")

    print("[*] Loading and splitting data...")
    meth_df, ent_df, ages_df = load_raw_real_data()
    feature_names = meth_df.columns.tolist()
    input_dim = len(feature_names)

    # Must match run_real.py exactly (same bins, same seed), otherwise the saved
    # weights get scored against a test set they may have trained on
    indices = np.arange(len(meth_df))
    ages_raw = ages_df['Age'].values
    age_bins = np.digitize(ages_raw, bins=np.percentile(ages_raw, [25, 50, 75]))

    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=age_bins)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=age_bins[temp_idx])

    # train/val loaders discarded; only here to refit the scalers on train
    _, _, test_loader, age_scaler = create_scaled_dataloaders(
        meth_df, ent_df, ages_df, train_idx, val_idx, test_idx
    )

    print(f"\n[*] Initializing DualStreamEntropyClockNet for {input_dim} loci...")
    model = DualStreamEntropyClockNet(input_dim=input_dim)

    if not os.path.exists(weight_file):
        print(f"[!] Error: Could not find '{weight_file}'")
        sys.exit(1)

    # map_location in case the weights were saved from MPS
    model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu'), weights_only=True))

    model.age_scaler = age_scaler
    model.eval()

    print(f"[+] Weights from '{weight_file}' loaded successfully!")

    print("\n[*] Evaluating saved model on held-out test set...")
    os.makedirs("outputs", exist_ok=True)

    with torch.no_grad():
        # eval_ prefix so this doesn't clobber the graphs from a run_real.py run
        evaluate_and_plot(model, test_loader, feature_names,
                          plot_path="outputs/eval_predicted_vs_actual.png",
                          attention_path="outputs/eval_attention_summary.png",
                          importance_path="outputs/eval_stream_importance.png")

    print("\n[+] Evaluation complete! Check the outputs/ folder for the 'eval_' graphs!")

if __name__ == "__main__":
    main()
