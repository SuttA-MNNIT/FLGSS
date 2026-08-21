"""
FedClustering Baseline (Zhao et al., 2025)
===========================================
Semantic Multi-Center Federated Clustering:
Maintains multiple cluster models on the server to capture multi-modal
client distributions in non-IID networks:
    Clients download the best matching cluster model, train locally,
    and server updates cluster centers via soft K-means clustering.
"""

from typing import Dict, List, Tuple
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainingConfig
from baselines.fedavg import BaselineClassifier


class FedClusteringServer:
    """
    FedClustering Server: Manages K cluster models.
    """

    def __init__(
        self,
        global_model: BaselineClassifier,
        num_clusters: int = 3,
    ):
        self.base_model = global_model
        self.num_clusters = num_clusters
        self.round_count = 0

        # K cluster model parameter sets
        init_params = global_model.get_head_params()
        self.cluster_models: List[Dict[str, torch.Tensor]] = []
        for i in range(num_clusters):
            c_params = {}
            for k, v in init_params.items():
                noise = torch.randn_like(v) * 0.02 if i > 0 else 0.0
                c_params[k] = (v + noise).clone()
            self.cluster_models.append(c_params)

        # Track client assignments
        self.client_cluster_map: Dict[int, int] = {}

    def get_cluster_for_client(self, client_id: int) -> int:
        return self.client_cluster_map.get(client_id, client_id % self.num_clusters)

    def aggregate(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor], int]],  # (client_id, params, n_samples)
    ):
        self.round_count += 1
        if not client_updates:
            return

        # Re-assign clients to closest cluster based on L2 distance
        cluster_groups: Dict[int, List[Tuple[Dict[str, torch.Tensor], int]]] = {
            c: [] for c in range(self.num_clusters)
        }

        for client_id, params, n_samples in client_updates:
            # Find closest cluster
            min_dist = float("inf")
            best_c = 0
            for c_idx, c_params in enumerate(self.cluster_models):
                dist = sum(
                    torch.norm(params[k].to(c_params[k].device) - c_params[k]).item()
                    for k in params
                )
                if dist < min_dist:
                    min_dist = dist
                    best_c = c_idx

            self.client_cluster_map[client_id] = best_c
            cluster_groups[best_c].append((params, n_samples))

        # Update each cluster model via weighted averaging
        for c_idx in range(self.num_clusters):
            group = cluster_groups[c_idx]
            if group:
                total_samples = sum(n for _, n in group)
                avg_params = {}
                for key in self.cluster_models[c_idx]:
                    ref_dev = self.cluster_models[c_idx][key].device
                    avg_params[key] = sum(
                        (n / total_samples) * p[key].to(ref_dev) for p, n in group
                    )
                self.cluster_models[c_idx] = avg_params

        # Update base model with weighted average of all cluster models for global evaluation
        target_dev = next(self.base_model.parameters()).device if list(self.base_model.parameters()) else torch.device("cpu")
        cluster_counts = [0] * self.num_clusters
        for c in self.client_cluster_map.values():
            cluster_counts[c] += 1
        total_assigned = max(sum(cluster_counts), 1)

        global_avg = {}
        for key in self.cluster_models[0]:
            global_avg[key] = sum(
                (cluster_counts[c_idx] / total_assigned) * self.cluster_models[c_idx][key].to(target_dev)
                for c_idx in range(self.num_clusters)
            )
        self.base_model.set_head_params(global_avg)

    def get_global_model(self) -> BaselineClassifier:
        return self.base_model


class FedClusteringClient:
    """
    FedClustering Client: Trains on the assigned cluster model.
    """

    def __init__(
        self,
        client_id: int,
        model: BaselineClassifier,
        train_config: TrainingConfig,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.model = model
        self.train_config = train_config
        self.device = device

    def local_train(
        self,
        dataloader: DataLoader,
        cluster_params: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        self.model.set_head_params(cluster_params)
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.head.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0
        num_samples = 0

        for epoch in range(self.train_config.local_epochs):
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                logits = self.model(batch_data)
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                if epoch == 0:
                    num_samples += len(batch_labels)

        avg_loss = total_loss / max(num_batches, 1)
        return self.model.get_head_params(), num_samples, avg_loss

    def get_comm_cost_bytes(self) -> int:
        return self.model.get_head_param_count() * 4 * 2
