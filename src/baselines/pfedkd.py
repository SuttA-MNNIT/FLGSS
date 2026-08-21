"""
PFedKD Baseline (Li et al., 2025)
=================================
Personalized Federated Knowledge Distillation:
Combines global consensus distillation with adaptive personalized local adaptation.
Each client maintains a personalized head that learns from both local private labels
and soft dark knowledge from the global ensemble:
    Loss = (1 - λ) * Loss_CE + λ * T² * KL(softmax(z_local/T), softmax(z_global/T))
"""

from typing import Dict, List, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import TrainingConfig
from baselines.fedavg import BaselineClassifier, FedAvgServer


class PFedKDClient:
    """
    PFedKD Client: Personalized student with global consensus distillation.
    """

    def __init__(
        self,
        client_id: int,
        model: BaselineClassifier,
        train_config: TrainingConfig,
        temperature: float = 2.0,
        lam: float = 0.5,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.model = model
        self.train_config = train_config
        self.temperature = temperature
        self.lam = lam
        self.device = device

        # Personalized local parameters (preserved across rounds)
        self.personalized_params = None

    def local_train(
        self,
        dataloader: DataLoader,
        global_params: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        # Global guide model
        global_model = copy.deepcopy(self.model)
        global_model.set_head_params(global_params)
        global_model.to(self.device)
        global_model.eval()

        # Initialize personalized model if first round, otherwise resume local state
        if self.personalized_params is None:
            self.model.set_head_params(global_params)
        else:
            # Blend global and personalized
            blended = {}
            for k in global_params:
                blended[k] = (1 - self.lam) * self.personalized_params[k].to(self.device) + self.lam * global_params[k].to(self.device)
            self.model.set_head_params(blended)

        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.head.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        ce_loss_fn = nn.CrossEntropyLoss()
        kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

        total_loss = 0.0
        num_batches = 0
        num_samples = 0

        for epoch in range(self.train_config.local_epochs):
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Local logits
                local_logits = self.model(batch_data)
                loss_ce = ce_loss_fn(local_logits, batch_labels)

                # Global guide logits
                with torch.no_grad():
                    guide_logits = global_model(batch_data)

                # Knowledge distillation loss
                p_s = F.log_softmax(local_logits / self.temperature, dim=-1)
                p_t = F.softmax(guide_logits / self.temperature, dim=-1)
                loss_kd = kl_loss_fn(p_s, p_t) * (self.temperature ** 2)

                loss = (1.0 - self.lam) * loss_ce + self.lam * loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                if epoch == 0:
                    num_samples += len(batch_labels)

        # Save personalized state
        current_params = self.model.get_head_params()
        self.personalized_params = {k: v.clone() for k, v in current_params.items()}

        avg_loss = total_loss / max(num_batches, 1)
        return current_params, num_samples, avg_loss

    def get_comm_cost_bytes(self) -> int:
        return self.model.get_head_param_count() * 4 * 2


PFedKDServer = FedAvgServer
