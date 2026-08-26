import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import all the custom modules you just built!
from src.entropy import calculate_metrics_from_reads
from src.data_loader import get_dataloaders
from src.model import DualStreamEntropyClockNet
from src.train import train_model
from src.evaluate import evaluate_model, plot_predictions, generate_shap_explanations

def generate_synthetic_reads(n_patients=200, n_loci=50, n_reads=30, n_cpgs=4):
    """
    Simulates read-level DNA sequencing data so we can test the pipeline today.
    Creates a hidden relationship with 'Age' so the neural network has something to learn.
    """
    print(f"Generating synthetic sequencing reads for {n_patients} patients...")
    # Generate random binary reads (0=unmethylated, 1=methylated)
    reads_array = np.random.randint(0, 2, size=(n_patients, n_loci, n_reads, n_cpgs))
    
    # Create a dummy "Age" variable (normally distributed around 50 years old)
    ages = np.random.normal(loc=50, scale=15, size=n_patients)
    
    # Inject a biological signal: artificially modify reads based on age
    # Older patients get more 'messy' patterns (entropy) and higher methylation at specific loci
    for i in range(n_patients):
        age_factor = ages[i] / 100.0 
        if age_factor > 0.5:
            # Inject higher methylation signal in the first 10 loci for older patients
            reads_array[i, :10, :, :] = np.random.choice([0, 1], p=[0.2, 0.8], size=(10, n_reads, n_cpgs))
            
    patient_ids = [f"Patient_{i}" for i in range(n_patients)]
    locus_ids = [f"cg_{j}" for j in range(n_loci)]
    
    return reads_array, ages, patient_ids, locus_ids

def main():
    # ---------------------------------------------------------
    # 1. DATA GENERATION & FEATURE EXTRACTION
    # ---------------------------------------------------------
    reads_array, ages, patient_ids, locus_ids = generate_synthetic_reads()
    
    print("\nExtracting Mean Methylation and Shannon Entropy from reads...")
    mean_meth_df, entropy_df = calculate_metrics_from_reads(reads_array, patient_ids, locus_ids)
    
    # Concatenate features horizontally: [Meth_features, Entropy_features]
    X_full = pd.concat([mean_meth_df, entropy_df], axis=1).values
    y_full = ages
    
    # Generate feature names for SHAP plots
    feature_names = [f"{loc}_Meth" for loc in locus_ids] + [f"{loc}_Ent" for loc in locus_ids]
    n_features_per_stream = mean_meth_df.shape[1]

    # ---------------------------------------------------------
    # 2. DATA SPLITTING & LEAK-FREE SCALING
    # ---------------------------------------------------------
    print("\nSplitting data into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    
    train_loader, test_loader, scaler = get_dataloaders(X_train, y_train, X_test, y_test, batch_size=16)

    # ---------------------------------------------------------
    # 3. BUILD THE UPGRADED NEURAL NETWORK
    # ---------------------------------------------------------
    print("\nInitializing the Dual-Stream Attention-Fusion Network...")
    model = DualStreamEntropyClockNet(n_features_per_stream=n_features_per_stream)

    # ---------------------------------------------------------
    # 4. TRAIN THE MODEL
    # ---------------------------------------------------------
    model = train_model(model, train_loader, test_loader, epochs=50, learning_rate=0.005)

    # ---------------------------------------------------------
    # 5. EVALUATE & INTERPRET
    # ---------------------------------------------------------
    print("\nEvaluating final model on the held-out Test Set...")
    # Convert scaled test data to tensors for evaluation
    X_test_tensor = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    X_train_tensor = torch.tensor(scaler.transform(X_train), dtype=torch.float32) # For SHAP background
    
    metrics = evaluate_model(model, X_test_tensor, y_test_tensor)
    print(f"Final Test MAE: {metrics['mae']:.2f} years | Pearson r: {metrics['pearson_r']:.3f}")
    
    print("\nGenerating Output Plots...")
    plot_predictions(metrics)
    generate_shap_explanations(model, X_train_tensor, X_test_tensor, feature_names)
    
    print("\nPipeline Complete! Check the 'outputs/' folder for your plots.")

if __name__ == "__main__":
    main()