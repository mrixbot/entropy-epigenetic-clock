import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error
import copy

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    """
    Trains the Dual-Stream network with Early Stopping and a dynamic Learning Rate Scheduler.
    """
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 1. Initialize the scheduler
    # Drops the learning rate by 70% if the validation MAE plateaus for 12 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=12)
    best_val_mae = float('inf')
    best_weights = None
    epochs_without_improvement = 0

    print(f"[*] Starting Training Loop (Max {epochs} Epochs, Early Stopping Patience={patience})...")
    print("-" * 80)
    print(f"{'EPOCH':<7} | {'LR':<10} | {'TRAIN LOSS':<12} | {'VAL MAE (YRS)':<15} | {'STATUS'}")
    print("-" * 80)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        # Track current learning rate for terminal output
        current_lr = optimizer.param_groups[0]['lr']

        for meth, ent, age in train_loader:
            optimizer.zero_grad()
            predictions = model(meth, ent)
            
            base_loss = criterion(predictions, age)
            
            # Dynamic weighting: Heavily penalize errors on outlier ages.
            weights = 1.0 + (torch.abs(age) * 2.0) 
            
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

        # 2. Step the scheduler based on current epoch's validation MAE
        #scheduler.step(val_mae)

        status = "progress"
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            status = "✨ NEW BEST"
        else:
            epochs_without_improvement += 1

        print(f"Ep {epoch+1:03d} | {current_lr:.6f} | Loss: {avg_train_loss:<8.3f} | Val MAE: {val_mae:<9.2f} yrs | {status}")

        if epochs_without_improvement >= patience:
            print(f"\n[!] Early stopping triggered at epoch {epoch+1}! Validation error stopped improving.")
            break

    print("-" * 80)
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"[+] Loaded best model weights corresponding to lowest VALIDATION error "
              f"(val MAE in scaled units: {best_val_mae:.3f}).")
        print("[i] Test set has not been touched yet -- final score comes from evaluate_and_plot().")

    return model