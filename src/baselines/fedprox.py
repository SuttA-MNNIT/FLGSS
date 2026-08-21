"""
FedProx Baseline (Li et al., MLSys 2020)
========================================
Federated Proximal optimization:
Mitigates client drift in non-IID settings by adding a proximal penalty term
to local optimization:
    Loss = CrossEntropy(w) + (μ/2) * ||w - w_global||²
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainingConfig
from baselines.fedavg import BaselineClassifier, FedAvgServer


class FedProxClient:
    """
    FedProx Client with proximal regularization.
    """

    def __init__(
        self,
        client_id: int,
        model: BaselineClassifier,
        train_config: TrainingConfig,
        mu: float = 0.01,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.model = model
        self.train_config = train_config
        self.mu = mu
        self.device = device

    def local_train(
        self,
        dataloader: DataLoader,
        global_params: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        self.model.set_head_params(global_params)
        self.model.to(self.device)
        self.model.train()

        global_tensors = {
            k: v.to(self.device) for k, v in global_params.items()
        }

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
                ce_loss = criterion(logits, batch_labels)

                # Proximal term: (μ/2) * ||w - w_global||²
                prox_term = 0.0
                for name, param in self.model.head.named_parameters():
                    if name in global_tensors:
                        prox_term += ((param - global_tensors[name]) ** 2).sum()
                prox_term = (self.mu / 2.0) * prox_term

                loss = ce_loss + prox_term

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


FedProxServer = FedAvgServer
