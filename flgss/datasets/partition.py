import numpy as np
from typing import List, Dict, Union
import torch
from torch.utils.data import Dataset, Subset

def partition_data_non_iid(dataset: Dataset, num_clients: int, alpha: float, seed: int = 42) -> List[np.ndarray]:
    """
    Partitions dataset indices among num_clients according to a Dirichlet distribution Dir(alpha).
    - alpha = 0.1: Extreme Non-IID skew (clients hold 1-2 dominant classes).
    - alpha = 1.0: Moderate Non-IID skew.
    - alpha = 10.0: Near-IID distribution.
    """
    np.random.seed(seed)
    
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    else:
        # Fallback for generic tensor datasets
        targets = np.array([y for _, y in dataset])

    num_classes = len(np.unique(targets))
    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        
        # Dirichlet proportions for class k
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Normalize to prevent rounding sum discrepancies
        proportions = proportions / proportions.sum()
        cuts = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        splits = np.split(idx_k, cuts)
        for i in range(num_clients):
            client_indices[i].extend(splits[i])

    # Convert to sorted int arrays
    client_partitions = [np.array(sorted(idx), dtype=int) for idx in client_indices]
    return client_partitions

class IndexedDataset(Dataset):
    """
    Wraps a dataset or subset to return (data, target, sample_index)
    Useful for high-speed tensor caching on H100 GPU.
    """
    def __init__(self, subset: Union[Dataset, Subset]):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, y, idx
