import torch
from src.data_loader import get_real_dataloaders
from src.model import DualStreamEntropyClockNet
from src.train import train_model
from src.evaluate import evaluate_and_plot

def main():
    print("==================================================")
    print(" ENTROPY CLOCK: REAL DATA PIPELINE (CHAN ET AL.)")
    print("==================================================")
    
    # 1. Load the real biological data (unpacking the age_scaler as the 5th return value)
    train_loader, test_loader, input_dim, feature_names, age_scaler = get_real_dataloaders()
    
    # 2. Initialize the architecture dynamically based on surviving loci
    print(f"\n[*] Initializing Dual-Stream Architecture for {input_dim} loci...")
    model = DualStreamEntropyClockNet(input_dim=input_dim)
    
    # 3. Train the model with Early Stopping
    print("\n[*] Commencing training loop on biological data...")
    trained_model = train_model(model, train_loader, test_loader, epochs=100, patience=15)
    
    # 4. Evaluate, un-scale predictions back to real years, and generate plots
    print("\n[*] Evaluating final model, un-scaling targets, and calculating SHAP values...")
    
    # Temporarily attach the scaler to the model object so evaluate_and_plot can use it
    trained_model.age_scaler = age_scaler
    
    evaluate_and_plot(trained_model, test_loader, feature_names, 
                      plot_path="outputs/real_predicted_vs_actual.png", 
                      shap_path="outputs/real_shap_summary.png")
                      
    print("\n[+] Real pipeline execution complete. Check the outputs/ folder!")

if __name__ == "__main__":
    main()