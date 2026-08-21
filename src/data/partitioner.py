"""
Non-IID Data Partitioner
=========================
Partitions a dataset across K federated clients:
1. Artificial non-IID: Latent Dirichlet Allocation (LDA) with concentration parameter α (CIFAR-10, UCI-HAR).
   - α = 0.1 : Extreme non-IID skew
   - α = 1.0 : Moderate non-IID skew
   - α = 10.0: Low skew / near-IID
2. Natural non-IID: Partitioned naturally by physical sensor/device ID (Intel Berkeley Lab, N-BaIoT).

Reference: Section V-A of the paper.
"""

from typing import Dict, List

import numpy as np

from data.datasets import get_targets


def dirichlet_partition(
    dataset,
    num_clients: int,
    alpha: float,
    num_classes: int,
    seed: int = 42,
    min_samples: int = 2,
) -> Dict[int, List[int]]:
    """
    Partition dataset indices across clients via Dirichlet allocation.
    """
    rng = np.random.default_rng(seed)
    targets = get_targets(dataset)

    # Group sample indices by class
    class_indices = {c: np.where(targets == c)[0] for c in range(num_classes)}

    # Initialize empty assignment for each client
    client_indices: Dict[int, List[int]] = {k: [] for k in range(num_clients)}

    for c in range(num_classes):
        indices_c = class_indices[c].copy()
        rng.shuffle(indices_c)

        # Sample proportions from Dirichlet(α, ..., α)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Ensure minimum allocation
        proportions = np.maximum(proportions, 1e-6)
        proportions /= proportions.sum()

        # Compute cumulative split points
        splits = (np.cumsum(proportions) * len(indices_c)).astype(int)
        splits = np.clip(splits, 0, len(indices_c))

        # Split indices according to proportions
        split_indices = np.split(indices_c, splits[:-1])

        for k in range(num_clients):
            if k < len(split_indices):
                client_indices[k].extend(split_indices[k].tolist())

    # Post-processing: ensure minimum samples per client
    all_indices = list(range(len(targets)))
    for k in range(num_clients):
        if len(client_indices[k]) < min_samples:
            extra = rng.choice(all_indices, size=min_samples, replace=False)
            client_indices[k].extend(extra.tolist())

    return client_indices


def natural_partition(
    dataset,
    num_clients: int = None,
) -> Dict[int, List[int]]:
    """
    Partition dataset naturally based on client_ids attribute (Intel Lab / N-BaIoT).
    """
    if hasattr(dataset, "client_ids"):
        client_ids = dataset.client_ids
        unique_clients = np.unique(client_ids)
        if num_clients is not None:
            unique_clients = unique_clients[:num_clients]
        
        partition = {}
        for idx, c_id in enumerate(unique_clients):
            match_indices = np.where(client_ids == c_id)[0].tolist()
            partition[idx] = match_indices
        return partition
    else:
        # Fallback to uniform split
        n_total = len(dataset)
        k = num_clients or 10
        indices = np.array_split(np.arange(n_total), k)
        return {i: idx.tolist() for i, idx in enumerate(indices)}


def partition_dataset(
    dataset,
    dataset_name: str,
    num_clients: int,
    alpha: float,
    num_classes: int,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Unified partition dispatcher based on dataset type.
    """
    if dataset_name in ("intel", "nbaiot"):
        return natural_partition(dataset, num_clients=num_clients)
    else:
        return dirichlet_partition(
            dataset,
            num_clients=num_clients,
            alpha=alpha,
            num_classes=num_classes,
            seed=seed,
        )


def get_client_class_distribution(
    dataset,
    client_indices: Dict[int, List[int]],
    num_classes: int,
) -> Dict[int, np.ndarray]:
    """
    Compute the class distribution for each client.
    """
    targets = get_targets(dataset)
    client_dist = {}
    for k, indices in client_indices.items():
        if len(indices) == 0:
            client_dist[k] = np.zeros(num_classes, dtype=int)
        else:
            counts = np.bincount(targets[indices], minlength=num_classes)
            client_dist[k] = counts
    return client_dist


def print_partition_summary(
    client_dist: Dict[int, np.ndarray],
    num_classes: int,
    top_k: int = 5,
):
    """Print a summary of the data partition across clients."""
    total_per_class = np.zeros(num_classes, dtype=int)
    total_per_client = []

    for k, counts in client_dist.items():
        total_per_class += counts
        total_per_client.append(counts.sum())

    total_per_client = np.array(total_per_client)

    print(f"\n{'='*60}")
    print(f"  Partition Summary: {len(client_dist)} clients, {num_classes} classes")
    print(f"{'='*60}")
    print(f"  Total samples: {total_per_client.sum()}")
    print(f"  Samples per client: min={total_per_client.min()}, "
          f"max={total_per_client.max()}, "
          f"mean={total_per_client.mean():.1f}")
    print(f"  Samples per class (global): {total_per_class}")

    skew = np.array([
        np.max(client_dist[k]) / max(np.sum(client_dist[k]), 1)
        for k in range(len(client_dist))
    ])
    most_skewed = np.argsort(-skew)[:top_k]
    print(f"\n  Most skewed clients (top {top_k}):")
    for k in most_skewed:
        dominant_class = np.argmax(client_dist[k])
        print(f"    Client {k}: {client_dist[k].sum()} samples, "
              f"dominant class={dominant_class} "
              f"({skew[k]*100:.1f}% concentration)")
    print(f"{'='*60}\n")
