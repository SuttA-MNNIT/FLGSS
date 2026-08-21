"""
MOON Baseline (Li et al., CVPR 2021)
====================================
Model-Contrastive Federated Learning:
Corrects local training via model-level contrastive learning in the representation space:
    Loss = Loss_CE + μ * Loss_Con
where Loss_Con maximizes cosine similarity between current representation z
and global model representation z_glob (positive pair) while minimizing similarity
with previous local model representation z_prev (negative pair).
"""

from typing import Dict, List, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import TrainingConfig
from baselines.fedavg import BaselineClassifier, FedAvgServer


class MOONClient:
    """
    MOON Client: Model-Contrastive Federated Learning.
    """

    def __init__(
        self,
        client_id: int,
        model: BaselineClassifier,
        train_config: TrainingConfig,
        mu: float = 0.1,
        temperature: float = 0.5,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.model = model
        self.train_config = train_config
        self.mu = mu
        self.temperature = temperature
        self.device = device

        # Previous round local model parameters
        self.prev_local_params = None

    def local_train(
        self,
        dataloader: DataLoader,
        global_params: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        # Global reference model
        global_model = copy.deepcopy(self.model)
        global_model.set_head_params(global_params)
        global_model.to(self.device)
        global_model.eval()

        # Previous local model (if available)
        prev_model = None
        if self.prev_local_params is not None:
            prev_model = copy.deepcopy(self.model)
            prev_model.set_head_params(self.prev_local_params)
            prev_model.to(self.device)
            prev_model.eval()

        # Current local model
        self.model.set_head_params(global_params)
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.head.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        ce_criterion = nn.CrossEntropyLoss()
        cos_sim = nn.CosineSimilarity(dim=-1)

        total_loss = 0.0
        num_batches = 0
        num_samples = 0

        for epoch in range(self.train_config.local_epochs):
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Current representation and logits
                z_curr = self.model.get_representation(batch_data)
                logits = self.model.head[2](z_curr)
                loss_ce = ce_criterion(logits, batch_labels)

                # Global representation (positive)
                with torch.no_grad():
                    z_glob = global_model.get_representation(batch_data)

                # Contrastive loss
                pos_sim = cos_sim(z_curr, z_glob) / self.temperature  # (B,)

                if prev_model is not None:
                    with torch.no_grad():
                        z_prev = prev_model.get_representation(batch_data)
                    neg_sim = cos_sim(z_curr, z_prev) / self.temperature  # (B,)
                    logits_con = torch.stack([pos_sim, neg_sim], dim=1)   # (B, 2)
                    labels_con = torch.zeros(z_curr.size(0), dtype=torch.long, device=self.device)
                    loss_con = ce_criterion(logits_con, labels_con)
                else:
                    loss_con = -torch.mean(pos_sim)

                loss = loss_ce + self.mu * loss_con

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                if epoch == 0:
                    num_samples += len(batch_labels)

        # Store current params as previous for next time this client is selected
        local_params = self.model.get_head_params()
        self.prev_local_params = {k: v.clone() for k, v in local_params.items()}

        avg_loss = total_loss / max(num_batches, 1)
        return local_params, num_samples, avg_loss

    def get_comm_cost_bytes(self) -> int:
        return self.model.get_head_param_count() * 4 * 2


MOONServer = FedAvgServer
