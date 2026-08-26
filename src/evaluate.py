import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import os

# ---------------------------------------------------------
# EVALUATION, METRICS, & STREAM IMPORTANCE PIPELINE
# ---------------------------------------------------------
def evaluate_and_plot(model, test_loader, feature_names, plot_path="outputs/plot.png", shap_path="outputs/shap.png"):
    model.eval()
    all_preds = []
    all_targets = []
    
    # Track attention weights to measure stream importance
    meth_weights_all = []
    ent_weights_all = []
    
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    print("[*] Running inference on test set and capturing attention weights...")
    with torch.no_grad():
        for meth, ent, age in test_loader:
            preds = model(meth, ent)
            all_preds.extend(preds.numpy().flatten())
            all_targets.extend(age.numpy().flatten())
            
            # --- Extract Gated Attention Weights Dynamically ---
            m_out = model.meth_stream(meth)
            e_out = model.ent_stream(ent)
            combined = torch.cat([m_out, e_out], dim=1)
            attn_weights = model.attention(combined)
            
            meth_weights_all.extend(attn_weights[:, 0].numpy())
            ent_weights_all.extend(attn_weights[:, 1].numpy())
            
    all_preds = np.array(all_preds).reshape(-1, 1)
    all_targets = np.array(all_targets).reshape(-1, 1)
    
    # Inverse transform to get back to real years if scaler is attached
    if hasattr(model, 'age_scaler'):
        all_preds = model.age_scaler.inverse_transform(all_preds).flatten()
        all_targets = model.age_scaler.inverse_transform(all_targets).flatten()
    else:
        all_preds = all_preds.flatten()
        all_targets = all_targets.flatten()
    
    # -------------------------
    # METRICS CALCULATION
    # -------------------------
    mae = mean_absolute_error(all_targets, all_preds)
    
    if len(np.unique(all_preds)) > 1:
        r, _ = pearsonr(all_targets, all_preds)
    else:
        r = 0.0
        
    r2 = r2_score(all_targets, all_preds)
    
    print("\n======================================")
    print("      FINAL MODEL PERFORMANCE")
    print("======================================")
    print(f"  Mean Absolute Error (MAE): {mae:.2f} years")
    print(f"  Pearson Correlation (r):   {r:.3f}")
    print(f"  R-Squared (R2):            {r2:.3f}")
    print("======================================\n")
    
    # -------------------------
    # REGRESSION PLOT
    # -------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(all_targets, all_preds, alpha=0.7, color="#2c3e50", edgecolors='white', s=60)
    
    min_val = min(all_targets.min(), all_preds.min())
    max_val = max(all_targets.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
    
    plt.xlabel("Actual Age (Years)", fontsize=12)
    plt.ylabel("Predicted Age (Years)", fontsize=12)
    plt.title(f"Dual-Stream Epigenetic Clock (MAE: {mae:.2f} yrs | r: {r:.2f})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Regression plot saved to {plot_path}")
    
    # ---------------------------------------------------------
    # STREAM IMPORTANCE BAR CHART (GATED ATTENTION ANALYSIS)
    # ---------------------------------------------------------
    print("[*] Generating Dual-Stream Attention Importance Chart...")
    try:
        avg_meth_weight = np.mean(meth_weights_all)
        avg_ent_weight = np.mean(ent_weights_all)
        
        streams = ['Methylation Stream', 'Shannon Entropy Stream']
        weights = [avg_meth_weight, avg_ent_weight]
        colors = ['#3498db', '#e74c3c']
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(streams, weights, color=colors, alpha=0.85, edgecolor='black', width=0.5)
        
        plt.ylabel("Mean Gated Attention Weight", fontsize=12)
        plt.title("Dual-Stream Biomarker Contribution Analysis", fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add exact percentages on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                     f"{height * 100:.1f}%",
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
            
        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Stream importance plot saved to {shap_path}")
        
    except Exception as e:
        print(f"\n[!] Warning: Stream importance plot failed. Error: {e}")
        
    print("\n[*] Pipeline complete!")