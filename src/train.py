import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error
import copy


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    """
    Trains the Dual-Stream network with Early Stopping to prevent overfitting.
    """
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_mae = float('inf')
    best_weights = None
    epochs_without_improvement = 0

    print(f"[*] Starting Training Loop (Max {epochs} Epochs, Early Stopping Patience={patience})...")
    print("-" * 65)
    print(f"{'EPOCH':<10} | {'TRAIN LOSS':<15} | {'VAL MAE (YEARS)':<15} | {'STATUS'}")
    print("-" * 65)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for meth, ent, age in train_loader:
            optimizer.zero_grad()
            predictions = model(meth, ent)
            
            # 1. Calculate unreduced MSE loss
            base_loss = criterion(predictions, age)
            
            # 2. Dynamic weighting: Heavily penalize errors on outlier ages.
            # NOTE (Bug 2 Fix/Note): 'age' here is standardized (z-scored), meaning values near 0 
            # represent the mean age, while outliers sit around +/- 1.5 to 2.0 standard deviations. 
            # This formula gives average ages a baseline weight multiplier of 1.0, 
            # scaling up to ~5.0 for extreme outlier age groups.
            weights = 1.0 + (torch.abs(age) * 2.0) 
            
            # 3. Apply weights and reduce
            loss = (base_loss * weights).mean()
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for meth, ent, age in val_loader:
                preds = model(meth, ent)
                val_preds.extend(preds.numpy().flatten())
                val_targets.extend(age.numpy().flatten())

        avg_train_loss = np.mean(train_losses)
        val_mae = mean_absolute_error(val_targets, val_preds)

        status = "progress"
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            status = "✨ NEW BEST"
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch+1:03d}  | Loss: {avg_train_loss:<10.3f}   | Val MAE: {val_mae:<8.2f} yrs   | {status}")

        if epochs_without_improvement >= patience:
            print(f"\n[!] Early stopping triggered at epoch {epoch+1}! Validation error stopped improving.")
            break

    print("-" * 65)
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"[+] Loaded best model weights corresponding to lowest VALIDATION error "
              f"(val MAE in scaled units: {best_val_mae:.3f}).")
        print("[i] Test set has not been touched yet -- final score comes from evaluate_and_plot().")

    return model