import random
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Import your exact modules
from src.data_loader import load_raw_real_data, create_scaled_dataloaders
from src.model import DualStreamEntropyClockNet
from src.evaluate import evaluate_and_plot

def main():
    # Set this to the name of the weight file you want to inspect
    weight_file = "weights/demo_locked_586_weights.pt" 

    print("==================================================")
    print(" ENTROPY CLOCK: STANDALONE WEIGHT EVALUATION")
    print("==================================================")
    
    # 1. Load Raw Data
    print("[*] Loading and splitting data...")
    meth_df, ent_df, ages_df = load_raw_real_data()
    feature_names = meth_df.columns.tolist()
    input_dim = len(feature_names)

    # 2. Replicate the Exact Stratified Split (Crucial for test set consistency)
    indices = np.arange(len(meth_df))
    ages_raw = ages_df['Age'].values
    age_bins = np.digitize(ages_raw, bins=np.percentile(ages_raw, [25, 50, 75]))
    
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=age_bins)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=age_bins[temp_idx])
    
    # 3. Build Dataloaders and Extract the Scaler
    _, _, test_loader, age_scaler = create_scaled_dataloaders(
        meth_df, ent_df, ages_df, train_idx, val_idx, test_idx
    )

    # 4. Initialize Architecture
    print(f"\n[*] Initializing DualStreamEntropyClockNet for {input_dim} loci...")
    model = DualStreamEntropyClockNet(input_dim=input_dim)
    
    # 5. Load Weights Safely
    if not os.path.exists(weight_file):
        print(f"[!] Error: Could not find '{weight_file}'")
        sys.exit(1)
        
    # We use map_location='cpu' just in case the weights were saved differently by MPS
    model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu'), weights_only=True))
    
    # 6. Attach Scaler (Required by your evaluate function)
    model.age_scaler = age_scaler
    model.eval()
    
    print(f"[+] Weights from '{weight_file}' loaded successfully!")
    
    # 7. Evaluate & Plot
    print("\n[*] Evaluating saved model on held-out test set...")
    os.makedirs("outputs", exist_ok=True)
    
    with torch.no_grad():
        # Saving with an 'eval_' prefix so you don't accidentally overwrite your main run graphs
        evaluate_and_plot(model, test_loader, feature_names,
                          plot_path="outputs/eval_predicted_vs_actual.png",
                          attention_path="outputs/eval_attention_summary.png",
                          importance_path="outputs/eval_stream_importance.png")
                          
    print("\n[+] Evaluation complete! Check the outputs/ folder for the 'eval_' graphs!")

if __name__ == "__main__":
    main()