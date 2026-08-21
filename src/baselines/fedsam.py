"""
FedSAM Baseline (Qu et al., ICML 2023)
======================================
Federated Sharpness-Aware Minimization:
Performs local Sharpness-Aware Minimization (SAM) on participating clients to seek
flat minima in the empirical loss landscape, mitigating non-IID generalization drop:
    min_w max_{||ε|| ≤ ρ} L_k(w + ε)
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainingConfig
from baselines.fedavg import BaselineClassifier, FedAvgServer


class FedSAMClient:
    """
    FedSAM Client: Uses Sharpness-Aware local optimization.
    """

    def __init__(
        self,
        client_id: int,
        model: BaselineClassifier,
        train_config: TrainingConfig,
        rho: float = 0.05,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.model = model
        self.train_config = train_config
        self.rho = rho
        self.device = device

    def local_train(
        self,
        dataloader: DataLoader,
        global_params: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        self.model.set_head_params(global_params)
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

                # Step 1: Compute standard gradient
                logits = self.model(batch_data)
                loss = criterion(logits, batch_labels)
                loss.backward()

                # Step 2: Compute SAM perturbation ε = ρ * ∇w / ||∇w||_2
                grad_norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2)
                        for p in self.model.head.parameters()
                        if p.grad is not None
                    ]),
                    p=2
                )

                scale = self.rho / (grad_norm + 1e-12)
                e_r = {}
                for p in self.model.head.parameters():
                    if p.grad is not None:
                        e = p.grad * scale
                        p.data.add_(e)
                        e_r[p] = e

                # Step 3: Compute gradient at perturbed point w + ε
                optimizer.zero_grad()
                logits_perturbed = self.model(batch_data)
                loss_perturbed = criterion(logits_perturbed, batch_labels)
                loss_perturbed.backward()

                # Restore weights: w + ε - ε = w
                for p, e in e_r.items():
                    p.data.sub_(e)

                # Step 4: Actual optimizer update using perturbed gradient
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                num_batches += 1
                if epoch == 0:
                    num_samples += len(batch_labels)

        avg_loss = total_loss / max(num_batches, 1)
        return self.model.get_head_params(), num_samples, avg_loss

    def get_comm_cost_bytes(self) -> int:
        return self.model.get_head_param_count() * 4 * 2


FedSAMServer = FedAvgServer
