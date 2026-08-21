"""
SCAFFOLD Baseline (Karimireddy et al., ICML 2020)
=================================================
Stochastic Controlled Averaging for Federated Learning:
Uses client-side control variates c_i and server-side control variates c
to correct for client drift caused by non-IID heterogeneity:
    Local update gradient: g_i(w) - c_i + c
"""

from typing import Dict, List, Tuple
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainingConfig
from baselines.fedavg import BaselineClassifier


class SCAFFOLDServer:
    """
    SCAFFOLD Server: Maintains global model weights and global control variate c.
    """

    def __init__(self, global_model: BaselineClassifier, num_clients: int):
        self.global_model = global_model
        self.num_clients = num_clients
        self.round_count = 0

        # Global control variate c
        self.c_global = {
            k: torch.zeros_like(v) for k, v in global_model.head.state_dict().items()
        }

    def aggregate(
        self,
        client_params_list: List[Dict[str, torch.Tensor]],
        delta_c_list: List[Dict[str, torch.Tensor]],
        client_weights: List[float],
    ):
        self.round_count += 1
        num_selected = len(client_params_list)
        if num_selected == 0:
            return

        total_weight = sum(client_weights)
        normalized = [w / total_weight for w in client_weights]

        # Update global model
        avg_params = {}
        for key in client_params_list[0]:
            ref_dev = next(self.global_model.parameters()).device if list(self.global_model.parameters()) else torch.device("cpu")
            avg_params[key] = sum(
                w * params[key].to(ref_dev) for w, params in zip(normalized, client_params_list)
            )
        self.global_model.set_head_params(avg_params)

        # Update global control variate c = c + (1/N) * Σ Δc_i
        for key in self.c_global:
            target_dev = self.c_global[key].device
            delta_sum = sum(dc[key].to(target_dev) for dc in delta_c_list)
            self.c_global[key] += (delta_sum / self.num_clients)

    def get_global_model(self) -> BaselineClassifier:
        return self.global_model

    def get_global_control_variate(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.c_global.items()}


class SCAFFOLDClient:
    """
    SCAFFOLD Client: Uses local control variates c_i and global c.
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

        # Local control variate c_i
        self.c_local = {
            k: torch.zeros_like(v).to(device)
            for k, v in model.head.state_dict().items()
        }

    def local_train(
        self,
        dataloader: DataLoader,
        global_params: Dict[str, torch.Tensor],
        c_global: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, float]:
        self.model.set_head_params(global_params)
        self.model.to(self.device)
        self.model.train()

        c_global_dev = {k: v.to(self.device) for k, v in c_global.items()}
        w_global_dev = {k: v.to(self.device) for k, v in global_params.items()}

        optimizer = torch.optim.Adam(
            self.model.head.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0
        num_samples = 0
        total_steps = 0

        for epoch in range(self.train_config.local_epochs):
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                logits = self.model(batch_data)
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()

                # Correct gradients with control variates: g_i - c_i + c
                with torch.no_grad():
                    for name, param in self.model.head.named_parameters():
                        if param.grad is not None and name in self.c_local:
                            param.grad += (c_global_dev[name] - self.c_local[name])

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                total_steps += 1
                if epoch == 0:
                    num_samples += len(batch_labels)

        # Compute updated local control variate c_i^+ and delta_c_i
        # c_i^+ = c_i - c + (1 / (K * η)) * (w_global - w_i)
        local_params = self.model.get_head_params()
        delta_c = {}
        eta = self.train_config.learning_rate
        scale = 1.0 / max(total_steps * eta, 1e-6)

        for name in local_params:
            w_diff = w_global_dev[name] - local_params[name].to(self.device)
            c_new = self.c_local[name] - c_global_dev[name] + scale * w_diff
            delta_c[name] = (c_new - self.c_local[name]).cpu()
            self.c_local[name] = c_new

        avg_loss = total_loss / max(num_batches, 1)
        return local_params, delta_c, num_samples, avg_loss

    def get_comm_cost_bytes(self) -> int:
        # Transmits model weights AND control variates in both directions
        head_bytes = self.model.get_head_param_count() * 4
        return head_bytes * 4  # (w + c) * 2 directions
