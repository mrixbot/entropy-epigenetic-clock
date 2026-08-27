import torch
from src.data_loader import get_synthetic_dataloaders
from src.model import DualStreamEntropyClockNet
from src.train import train_model
from src.evaluate import evaluate_and_plot


def main():
    print("==================================================")
    print(" ENTROPY CLOCK: SYNTHETIC PROOF OF CONCEPT")
    print("==================================================")

    # 1. Load the synthetic dummy data -- now three isolated loaders instead of two
    train_loader, val_loader, test_loader, input_dim, feature_names = get_synthetic_dataloaders()

    # 2. Initialize the architecture
    print(f"\n[*] Initializing Dual-Stream Architecture for {input_dim} loci...")
    model = DualStreamEntropyClockNet(input_dim=input_dim)

    # 3. Train the model -- checkpoint selection uses val_loader, not test_loader
    print("\n[*] Commencing training loop on synthetic data...")
    trained_model = train_model(model, train_loader, val_loader, epochs=50)

    # 4. Evaluate on the held-out test set
    print("\n[*] Evaluating final model on held-out test set...")
    evaluate_and_plot(trained_model, test_loader, feature_names,
                       plot_path="outputs/synthetic_predicted_vs_actual.png",
                       shap_path="outputs/synthetic_shap_summary.png")

    print("\n[+] Synthetic pipeline execution complete. Check the outputs/ folder!")


if __name__ == "__main__":
    main()
