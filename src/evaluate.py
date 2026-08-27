import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import os

# ---------------------------------------------------------
# STREAM-LEVEL PERMUTATION & ATTENTION EVALUATION
# ---------------------------------------------------------
def evaluate_and_plot(model, test_loader, feature_names, 
                      plot_path="outputs/real_predicted_vs_actual.png", 
                      attention_path="outputs/real_attention_summary.png",
                      importance_path="outputs/real_stream_importance.png",
                      n_sample_narratives=3):
    
    model.eval()
    all_preds, all_targets = [], []
    meth_weights_all, ent_weights_all = [], []
    
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    print("[*] Running inference on test set and capturing attention dynamics...")
    with torch.no_grad():
        for meth, ent, age in test_loader:
            preds = model(meth, ent)
            all_preds.extend(preds.numpy().flatten())
            all_targets.extend(age.numpy().flatten())
            
            # Extract Gated Attention Weights Dynamically
            m_out = model.meth_stream(meth)
            e_out = model.ent_stream(ent)
            combined = torch.cat([m_out, e_out], dim=1)
            attn_weights = model.attention(combined)
            
            meth_weights_all.extend(attn_weights[:, 0].numpy())
            ent_weights_all.extend(attn_weights[:, 1].numpy())
            
    all_preds = np.array(all_preds).reshape(-1, 1)
    all_targets = np.array(all_targets).reshape(-1, 1)
    
    # Inverse transform to get back to real years (explicit 2D reshape for scaler safety)
    if hasattr(model, 'age_scaler'):
        all_preds = model.age_scaler.inverse_transform(all_preds).flatten()
        all_targets = model.age_scaler.inverse_transform(all_targets).flatten()
    else:
        all_preds = all_preds.flatten()
        all_targets = all_targets.flatten()
    
    # -------------------------
    # 1. METRICS CALCULATION
    # -------------------------
    mae = mean_absolute_error(all_targets, all_preds)
    r, _ = pearsonr(all_targets, all_preds) if len(np.unique(all_preds)) > 1 else (0.0, 0)
    r2 = r2_score(all_targets, all_preds)
    
    print("\n======================================")
    print("      FINAL MODEL PERFORMANCE")
    print("======================================")
    print(f"  Mean Absolute Error (MAE): {mae:.2f} years")
    print(f"  Pearson Correlation (r):   {r:.3f}")
    print(f"  R-Squared (R2):            {r2:.3f}")
    print("======================================\n")
    
    # -------------------------
    # 2. REGRESSION SCATTER PLOT
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
    # 3. STREAM IMPORTANCE BAR CHART (ATTENTION)
    # ---------------------------------------------------------
    try:
        avg_meth_weight = np.mean(meth_weights_all)
        avg_ent_weight = np.mean(ent_weights_all)
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(['Methylation Stream', 'Shannon Entropy Stream'], 
                       [avg_meth_weight, avg_ent_weight], 
                       color=['#3498db', '#e74c3c'], alpha=0.85, edgecolor='black', width=0.5)
        
        plt.ylabel("Mean Gated Attention Weight", fontsize=12)
        plt.title("Dual-Stream Biomarker Contribution Analysis", fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.grid(True, axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                     f"{height * 100:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
            
        plt.savefig(attention_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Stream importance plot saved to {attention_path}")
    except Exception as e:
        print(f"\n[!] Warning: Stream importance plot failed. Error: {e}")

    # ---------------------------------------------------------
    # 4. STREAM-LEVEL PERMUTATION IMPORTANCE (ROBUST EXPLAINABILITY)
    # ---------------------------------------------------------
    print("[*] Calculating Stream-Level Permutation Importance...")
    try:
        m_list, e_list, a_list = [], [], []
        for m, e, a in test_loader:
            m_list.append(m)
            e_list.append(e)
            a_list.append(a)
        m_tensor = torch.cat(m_list, dim=0)
        e_tensor = torch.cat(e_list, dim=0)
        a_tensor = torch.cat(a_list, dim=0)

        with torch.no_grad():
            base_preds = model(m_tensor, e_tensor)
            preds_2d = base_preds.numpy().reshape(-1, 1)
            targets_2d = a_tensor.numpy().reshape(-1, 1)
            
            if hasattr(model, 'age_scaler'):
                base_p_inv = model.age_scaler.inverse_transform(preds_2d).flatten()
                base_t_inv = model.age_scaler.inverse_transform(targets_2d).flatten()
            else:
                base_p_inv = preds_2d.flatten()
                base_t_inv = targets_2d.flatten()
            base_mae = mean_absolute_error(base_t_inv, base_p_inv)

        # Permute Methylation Stream
        m_permuted = m_tensor[torch.randperm(len(m_tensor))]
        with torch.no_grad():
            perm_m_preds = model(m_permuted, e_tensor)
            perm_m_2d = perm_m_preds.numpy().reshape(-1, 1)
            perm_m_inv = model.age_scaler.inverse_transform(perm_m_2d).flatten() if hasattr(model, 'age_scaler') else perm_m_2d.flatten()
            mae_m_drop = mean_absolute_error(base_t_inv, perm_m_inv) - base_mae

        # Permute Entropy Stream
        e_permuted = e_tensor[torch.randperm(len(e_tensor))]
        with torch.no_grad():
            perm_e_preds = model(m_tensor, e_permuted)
            perm_e_2d = perm_e_preds.numpy().reshape(-1, 1)
            perm_e_inv = model.age_scaler.inverse_transform(perm_e_2d).flatten() if hasattr(model, 'age_scaler') else perm_e_2d.flatten()
            mae_e_drop = mean_absolute_error(base_t_inv, perm_e_inv) - base_mae

        plt.figure(figsize=(8, 6))
        streams = ['Methylation Stream (Permuted)', 'Shannon Entropy Stream (Permuted)']
        drops = [max(0, mae_m_drop), max(0, mae_e_drop)]
        bars = plt.bar(streams, drops, color=['#2980b9', '#c0392b'], alpha=0.85, edgecolor='black', width=0.5)
        
        plt.ylabel("Increase in MAE (Error Spike upon Shuffling)", fontsize=12)
        plt.title("Stream-Level Permutation Importance", fontsize=14, fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f"+{height:.2f} yrs", ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Permutation importance plot saved to {importance_path}")

    except Exception as e:
        print(f"\n[!] Warning: Permutation importance failed. Error: {e}")

    # ---------------------------------------------------------
    # 5. PRINT PATIENT-SPECIFIC ATTENTION BREAKDOWN NARRATIVE
    # ---------------------------------------------------------
    print("\n---------------------------------------------------------")
    print(f" SAMPLE PATIENT-SPECIFIC EXPLAINABILITY BREAKDOWN (TEST SET)")
    print("---------------------------------------------------------")
    for i in range(min(n_sample_narratives, len(all_preds))):
        act = all_targets[i]
        pred = all_preds[i]
        m_wt = meth_weights_all[i] * 100
        e_wt = ent_weights_all[i] * 100
        print(f" Patient {i+1} -> Actual Age: {act:.1f} yrs | Predicted: {pred:.1f} yrs")
        print(f"              -> Model Trust Split: {m_wt:.1f}% Methylation / {e_wt:.1f}% Entropy")
    print("---------------------------------------------------------\n")
    
    print("[*] Pipeline complete!")