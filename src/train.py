import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error
import copy


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    """
    Trains the Dual-Stream network with Early Stopping to prevent overfitting.

    CHANGED: the second argument is now val_loader, not test_loader. This
    function was previously selecting "best" weights based on performance
    on the SAME data later reported as the final test score -- meaning the
    reported MAE was not a genuine held-out result. Pass your validation
    split here; keep test_loader completely separate and only hand it to
    evaluate_and_plot() after this function returns.
    """
    criterion = nn.MSELoss()
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
            loss = criterion(predictions, age)
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
