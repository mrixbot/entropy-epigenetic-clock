import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

class EpigeneticDataset(Dataset):
    """
    Custom PyTorch Dataset for holding our tabular epigenetic features and chronological ages.
    """
    def __init__(self, features, ages):
        # Convert numpy arrays to PyTorch tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        # Ages need to be a 2D tensor for PyTorch regression -> Shape: (N, 1)
        self.ages = torch.tensor(ages, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        # Tells the DataLoader how many total patients we have
        return len(self.features)

    def __getitem__(self, idx):
        # Retrieves one patient's features and their actual age
        return self.features[idx], self.ages[idx]


def get_dataloaders(X_train, y_train, X_test, y_test, batch_size=16):
    """
    Takes raw numpy arrays, scales them leak-free, and returns PyTorch DataLoaders.
    """
    # 1. Initialize the normalizer
    scaler = StandardScaler()

    # 2. FIT strictly on training data, then transform training data
    # This prevents the model from knowing the average values of the test set
    X_train_scaled = scaler.fit_transform(X_train)

    # 3. Transform test/val data using the EXACT parameters learned from training
    # NEVER call .fit() on test data!
    X_test_scaled = scaler.transform(X_test)

    # 4. Wrap the scaled data in our custom PyTorch Dataset
    train_dataset = EpigeneticDataset(X_train_scaled, y_train)
    test_dataset = EpigeneticDataset(X_test_scaled, y_test)

    # 5. Create DataLoaders
    # shuffle=True for train so the model doesn't memorize the patient order
    # shuffle=False for test to keep our final predictions perfectly aligned
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler