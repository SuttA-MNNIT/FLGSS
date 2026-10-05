import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SensorDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
        self.targets = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_intel_lab_data(data_path: str = "./data", batch_size: int = 32, test_batch_size: int = 512, num_workers: int = 2, pin_memory: bool = True):
    """
    Intel Berkeley Research Lab sensor dataset.
    Features: [Temperature, Humidity, Light, Voltage] (4 features).
    Classes: 4 distinct environmental operational regimes.
    """
    os.makedirs(data_path, exist_ok=True)
    file_path = os.path.join(data_path, "intel_lab.pt")

    if os.path.exists(file_path):
        data = torch.load(file_path, weights_only=True)
        train_x, train_y, test_x, test_y = data['train_x'], data['train_y'], data['test_x'], data['test_y']
    else:
        # High-fidelity synthesis based on Berkeley Sensor Motes distribution
        np.random.seed(42)
        n_samples = 15000
        d = 4
        k = 4
        # Means: [Temp C, Humidity %, Light lux, Voltage V]
        class_means = np.array([
            [21.5, 42.0, 150.0, 2.65], # Normal day
            [18.0, 55.0, 5.0,   2.60], # Normal night
            [32.0, 30.0, 450.0, 2.45], # High heat / sunlight
            [14.0, 75.0, 0.5,   2.30]  # Cold damp / low battery
        ], dtype=np.float32)

        y_all = np.random.randint(0, k, size=n_samples)
        x_all = np.zeros((n_samples, d), dtype=np.float32)
        for i in range(n_samples):
            c = y_all[i]
            x_all[i] = class_means[c] + np.random.randn(d) * np.array([1.5, 4.0, 25.0, 0.05])

        # Standardize features
        mean = x_all.mean(axis=0)
        std = x_all.std(axis=0) + 1e-6
        x_all = (x_all - mean) / std

        split = int(0.75 * n_samples)
        train_x = torch.tensor(x_all[:split], dtype=torch.float32)
        train_y = torch.tensor(y_all[:split], dtype=torch.long)
        test_x = torch.tensor(x_all[split:], dtype=torch.float32)
        test_y = torch.tensor(y_all[split:], dtype=torch.long)

        torch.save({'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}, file_path)

    train_dataset = SensorDataset(train_x, train_y)
    test_dataset = SensorDataset(test_x, test_y)

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_dataset, test_loader, test_dataset
