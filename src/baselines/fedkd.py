"""
FedKD Baseline (Wu et al., 2022)
=================================
Federated Knowledge Distillation:
Applies mutual and mentor-student distillation on clients.
Clients train a student model with local supervision + KL divergence
matching the global teacher representation:
    Loss = (1 - α) * Loss_CE(y, ŷ) + α * T² * KL(softmax(z_s/T), softmax(z_t/T))
"""

from typing import Dict, List, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import TrainingConfig
from baselines.fedavg import BaselineClassifier, FedAvgServer


class FedKDClient:
    """
    FedKD Client: Knowledge distillation guided local training.
    """

    def __init__(
        self,
        client_id: int,
        model: BaselineClassifier,
        train_config: TrainingConfig,
        temperature: float = 2.0,
        alpha: float = 0.5,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.model = model
        self.train_config = train_config
        self.temperature = temperature
        self.alpha = alpha
        self.device = device

    def local_train(
        self,
        dataloader: DataLoader,
        global_params: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        # Teacher model (frozen global model)
        teacher_model = copy.deepcopy(self.model)
        teacher_model.set_head_params(global_params)
        teacher_model.to(self.device)
        teacher_model.eval()

        # Student model (initialized with global weights)
        self.model.set_head_params(global_params)
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

                # Student logits
                student_logits = self.model(batch_data)
                loss_ce = ce_loss_fn(student_logits, batch_labels)

                # Teacher soft logits
                with torch.no_grad():
                    teacher_logits = teacher_model(batch_data)

                # KL divergence distillation loss
                p_s = F.log_softmax(student_logits / self.temperature, dim=-1)
                p_t = F.softmax(teacher_logits / self.temperature, dim=-1)
                loss_kd = kl_loss_fn(p_s, p_t) * (self.temperature ** 2)

                loss = (1.0 - self.alpha) * loss_ce + self.alpha * loss_kd

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


FedKDServer = FedAvgServer
