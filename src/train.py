import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, test_loader, epochs=100, learning_rate=0.001):
    """
    The training loop for the epigenetic clock.
    """
    # 1. The "How Wrong Am I?" function
    # MSELoss (Mean Squared Error) heavily penalizes guesses that are wildly off.
    criterion = nn.MSELoss()
    
    # 2. The "Weight Nudger" (Optimizer)
    # Adam is an advanced gradient descent algorithm that updates the model's weights.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Put the model in training mode (turns on Dropout)
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_ages in train_loader:
            # Step A: Clear old nudges (reset gradients)
            optimizer.zero_grad()
            
            # Step B: Make a guess (Forward pass)
            predictions = model(batch_features)
            
            # Step C: Calculate how wrong the guess was (Calculate Loss)
            loss = criterion(predictions, batch_ages)
            
            # Step D: Figure out which direction to nudge the weights (Backward pass)
            loss.backward()
            
            # Step E: Actually nudge the weights (Optimizer step)
            optimizer.step()
            
            train_loss += loss.item()
            
        # Calculate average loss for this epoch
        avg_train_loss = train_loss / len(train_loader)
        
        # Every 10 epochs, let's take a "practice quiz" on the test set
        if (epoch + 1) % 10 == 0:
            # Put the model in evaluation mode (turns off Dropout)
            model.eval()
            test_loss = 0.0
            
            # torch.no_grad() tells PyTorch NOT to learn/nudge weights right now
            with torch.no_grad():
                for batch_features, batch_ages in test_loader:
                    predictions = model(batch_features)
                    # We use L1Loss (Mean Absolute Error) here because it is easier 
                    # for humans to read: "How many years off are we?"
                    mae_loss = nn.L1Loss()(predictions, batch_ages)
                    test_loss += mae_loss.item()
            
            avg_test_mae = test_loss / len(test_loader)
            print(f"Epoch [{epoch+1}/{epochs}] | Train MSE: {avg_train_loss:.2f} | Test MAE: {avg_test_mae:.2f} years")

    print("Training complete!")
    return model