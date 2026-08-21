"""
FedAvg Baseline (McMahan et al., 2017)
======================================
Standard Federated Averaging:
Each client trains a local model via Adam/SGD, and the server computes a weighted
average of client model parameters.

Architecture Fairness (Section V-B):
All vision and sensor baselines share the exact same frozen Anchor backbone as FLGSS.
"""

from typing import Dict, List, Tuple
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainingConfig


class BaselineClassifier(nn.Module):
    """
    Downstream classification head for all fair federated baselines.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Trainable classification head
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features from backbone or raw feature tensor."""
        if self.backbone is not None:
            with torch.no_grad():
                feat = self.backbone(x)
                if feat.dim() > 2:
                    feat = feat.view(feat.size(0), -1)
                return feat
        else:
            if x.dim() > 2:
                return x.view(x.size(0), -1)
            return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        logits = self.head(feat)
        return logits

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate representation for contrastive / KD methods."""
        feat = self.extract_features(x)
        # Representation from first layer of head
        h = self.head[0](feat)
        h = self.head[1](h)
        return h

    def get_head_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.head.state_dict().items()}

    def set_head_params(self, params: Dict[str, torch.Tensor]):
        self.head.load_state_dict(params)

    def get_head_param_count(self) -> int:
        return sum(p.numel() for p in self.head.parameters())


class FedAvgServer:
    """
    FedAvg Server: Performs weighted averaging of client head parameters.
    """

    def __init__(self, global_model: BaselineClassifier):
        self.global_model = global_model
        self.round_count = 0

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        client_weights: List[float],
    ):
        self.round_count += 1
        total_weight = sum(client_weights)
        if total_weight == 0:
            return

        normalized = [w / total_weight for w in client_weights]
        target_dev = next(self.global_model.parameters()).device if list(self.global_model.parameters()) else torch.device("cpu")
        avg_params = {}
        for key in client_params[0]:
            avg_params[key] = sum(
                w * params[key].to(target_dev) for w, params in zip(normalized, client_params)
            )

        self.global_model.set_head_params(avg_params)

    def get_global_model(self) -> BaselineClassifier:
        return self.global_model


class FedAvgClient:
    """
    FedAvg Client: Standard local cross-entropy optimization.
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
        return self.model.get_head_param_count() * 4 * 2  # upload + download
