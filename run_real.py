import torch
import numpy as np
import random
#torch.manual_seed(42)
#np.random.seed(42)
#random.seed(42)
from sklearn.model_selection import train_test_split
import os

from src.data_loader import load_raw_real_data, create_scaled_dataloaders
from src.model import DualStreamEntropyClockNet
from src.train import train_model
from src.evaluate import evaluate_and_plot

def main():
    print("==================================================")
    print(" ENTROPY CLOCK: FINAL STRATIFIED & LEAK-FREE PIPELINE")
    print("==================================================")

    # 1. Load Raw Data
    meth_df, ent_df, ages_df = load_raw_real_data()
    feature_names = meth_df.columns.tolist()
    input_dim = len(feature_names)

    # 2. Stratified Split on raw indices
    indices = np.arange(len(meth_df))
    ages_raw = ages_df['Age'].values
    age_bins = np.digitize(ages_raw, bins=np.percentile(ages_raw, [25, 50, 75]))
    
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=age_bins)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=age_bins[temp_idx])
    
    print(f"[*] Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    # 3. Create perfectly isolated, scaled dataloaders
    train_loader, val_loader, test_loader, age_scaler = create_scaled_dataloaders(
        meth_df, ent_df, ages_df, train_idx, val_idx, test_idx
    )

    # 4. Initialize and train
    print(f"\n[*] Initializing Dual-Stream Architecture for {input_dim} loci...")
    model = DualStreamEntropyClockNet(input_dim=input_dim)
    
    trained_model = train_model(model, train_loader, val_loader, epochs=100, patience=15)
    trained_model.age_scaler = age_scaler
    
    # 5. Evaluate & Plot (Using clean stream-level permutation importance)
    print("\n[*] Evaluating final model on held-out test set...")
    os.makedirs("outputs", exist_ok=True)
    evaluate_and_plot(trained_model, test_loader, feature_names,
                      plot_path="outputs/real_predicted_vs_actual.png",
                      attention_path="outputs/real_attention_summary.png",
                      importance_path="outputs/real_stream_importance.png")
                      
    torch.save(trained_model.state_dict(), "outputs/best_model_weights.pt")
    print("\n[+] Final model weights saved to outputs/best_model_weights.pt")
    print("[+] Pipeline complete! Check the outputs/ folder!")

if __name__ == "__main__":
    main()