import torch
from src.data_loader import get_real_dataloaders
from src.model import DualStreamEntropyClockNet
from src.train import train_model
from src.evaluate import evaluate_and_plot


def main():
    print("==================================================")
    print(" ENTROPY CLOCK: REAL DATA PIPELINE (CHAN ET AL.)")
    print("==================================================")

    # 1. Load the real biological data -- now three isolated loaders instead of two
    train_loader, val_loader, test_loader, input_dim, feature_names, age_scaler = get_real_dataloaders()

    # 2. Initialize the architecture dynamically based on surviving loci
    print(f"\n[*] Initializing Dual-Stream Architecture for {input_dim} loci...")
    model = DualStreamEntropyClockNet(input_dim=input_dim)

    # 3. Train the model with Early Stopping -- checkpoint selection now uses
    #    val_loader, NOT test_loader. test_loader is not referenced at all
    #    until step 4, below.
    print("\n[*] Commencing training loop on biological data...")
    trained_model = train_model(model, train_loader, val_loader, epochs=100, patience=15)

    # 4. Evaluate on the held-out TEST set -- the first and only time it's used
    print("\n[*] Evaluating final model on held-out test set, un-scaling targets...")

    trained_model.age_scaler = age_scaler

    evaluate_and_plot(trained_model, test_loader, feature_names,
                       plot_path="outputs/real_predicted_vs_actual.png",
                       shap_path="outputs/real_shap_summary.png")

    print("\n[+] Real pipeline execution complete. Check the outputs/ folder!")


if __name__ == "__main__":
    main()
