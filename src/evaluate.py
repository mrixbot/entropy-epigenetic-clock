import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import shap
import os

def evaluate_model(model, X_test_tensor, y_test_tensor):
    """
    Evaluates the model on the held-out test set and calculates MAE and Pearson r.
    """
    model.eval()
    # Turn off the gradient tracker to save memory since we aren't training anymore
    with torch.no_grad():
        predictions = model(X_test_tensor).squeeze().cpu().numpy()
    
    actuals = y_test_tensor.squeeze().cpu().numpy()
    
    # Calculate Mean Absolute Error (MAE)
    mae = float(np.mean(np.abs(predictions - actuals)))
    # Calculate Pearson correlation coefficient (r)
    r, p_value = pearsonr(predictions, actuals)
    
    return {
        "mae": mae, 
        "pearson_r": r, 
        "p_value": p_value, 
        "predictions": predictions, 
        "actuals": actuals
    }

def plot_predictions(metrics, output_dir="outputs"):
    """
    Generates a scatter plot of Predicted vs Actual ages.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    # Plot the individual patients
    plt.scatter(metrics["actuals"], metrics["predictions"], alpha=0.7, color='#2b6cb0')
    
    # Plot the ideal 1:1 line (Perfect Prediction)
    min_val = min(min(metrics["actuals"]), min(metrics["predictions"]))
    max_val = max(max(metrics["actuals"]), max(metrics["predictions"]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction Line")
    
    plt.title(f"Predicted vs Actual Age\nMAE: {metrics['mae']:.2f} Years | Pearson r: {metrics['pearson_r']:.3f}")
    plt.xlabel("Actual Chronological Age (Years)")
    plt.ylabel("Predicted Biological Age (Years)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = os.path.join(output_dir, "predicted_vs_actual.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved prediction plot to {save_path}")

def generate_shap_explanations(model, X_train_tensor, X_test_tensor, feature_names, output_dir="outputs"):
    """
    Generates SHAP values using the training set as the background to explain the test set.
    Uses GradientExplainer to properly handle the non-linear multiplicative Gated Fusion layer.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # 1. Use a random background sample from the training data
    n_background = min(100, X_train_tensor.shape[0])
    background_idx = np.random.choice(X_train_tensor.shape[0], size=n_background, replace=False)
    background = X_train_tensor[background_idx]
    
    # 2. Initialize the GradientExplainer (Robust to Attention/Gating mechanisms)
    explainer = shap.GradientExplainer(model, background)
    
    # 3. Calculate SHAP values for the unseen test set
    print("Calculating SHAP values (this might take a moment)...")
    shap_values = explainer.shap_values(X_test_tensor)
    
    # PyTorch GradientExplainer returns a list or tensor depending on version; handle extraction
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # If shape is (samples, features, 1), squeeze the last dimension
        shap_values = shap_values[:, :, 0]
        
    # 4. Generate and save the Summary Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_tensor.cpu().numpy(), feature_names=feature_names, show=False)
    
    save_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved SHAP summary plot to {save_path}")