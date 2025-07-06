import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

class LidDrivenDataset:
    """
    Handles loading of Lid Driven Cavity problem data from .npz files
    and provides dataloaders for training and validation sets.
    """
    def __init__(self, file_path_x_train, file_path_x_valid,
                 file_path_y_train, file_path_y_valid, transform=None):
        """
        Initializes the dataset from given training and validation files.
        """
        # Load data
        x_train = np.load(file_path_x_train)
        y_train = np.load(file_path_y_train)
        x_valid = np.load(file_path_x_valid)
        y_valid = np.load(file_path_y_valid)

        # Convert to tensors and select channels
        self.x_train = torch.tensor(x_train[:, [0, 2]], dtype=torch.float32)
        self.y_train = torch.tensor(y_train[:, [0, 1, 2]], dtype=torch.float32)
        self.x_valid = torch.tensor(x_valid[:, [0, 2]], dtype=torch.float32)
        self.y_valid = torch.tensor(y_valid[:, [0, 1, 2]], dtype=torch.float32)

        self.transform = transform

    def create_dataloader(self, batch_size=100, shuffle=True):
        """
        Creates DataLoaders for train and validation sets.

        Returns:
            train_loader, val_loader: PyTorch DataLoaders
        """
        if self.transform:
            self.x_train = self.transform(self.x_train)
            self.x_valid = self.transform(self.x_valid)

        train_dataset = TensorDataset(self.x_train, self.y_train)
        val_dataset = TensorDataset(self.x_valid, self.y_valid)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
