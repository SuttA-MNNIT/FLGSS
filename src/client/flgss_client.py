"""
FLGSS Client — On-Device Computation (Phases 1 & 2)
=====================================================
Implements the client-side logic from Algorithm 1, Steps A:

Phase 1 — Adapter Distillation (Calibration):
    Train a lightweight adapter A_φ to mimic the frozen Anchor E_A
    via knowledge distillation: minimize ||E_A(x) - A_φ(x)||² (Eq. 3).

Phase 2 — Statistical Abstraction:
    Project local data into the semantic space, group by class,
    and compute per-class empirical mean μ_{k,c} and covariance Σ_{k,c} (Eq. 4).
    Package the lightweight payload for transmission to the edge server.

Reference: Section IV-B, IV-C of the paper.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainingConfig, ModelConfig, PrivacyConfig
from models.adapter import AdapterNetwork
from privacy.differential_privacy import apply_dp_to_statistics


@dataclass
class ClientPayload:
    """
    The lightweight statistical payload transmitted from a client to its
    edge server. Contains per-class statistics.
    Communication cost: O(C × (d + d²/2)) vs O(|w|) for FedAvg.
    """
    client_id: int
    # Per-class statistics: class_label → (mean, covariance, count)
    class_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor, int]]
    total_samples: int


class FLGSSClient:
    """
    A single FLGSS client device.
    """

    def __init__(
        self,
        client_id: int,
        adapter: Optional[AdapterNetwork],
        train_config: TrainingConfig,
        model_config: ModelConfig,
        privacy_config: PrivacyConfig,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.adapter = adapter
        self.train_config = train_config
        self.model_config = model_config
        self.privacy_config = privacy_config
        self.device = device

        if self.adapter is not None:
            self.optimizer = torch.optim.Adam(
                self.adapter.parameters(),
                lr=train_config.learning_rate,
                weight_decay=train_config.weight_decay,
            )
            self.distillation_loss = nn.MSELoss()

    def distill_adapter(
        self,
        anchor: nn.Module,
        dataloader: DataLoader,
    ) -> float:
        """
        Phase 1: Train the local adapter via knowledge distillation from Anchor.
        """
        if self.adapter is None:
            return 0.0

        self.adapter.train()
        anchor.eval()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.train_config.local_epochs):
            epoch_loss = 0.0
            for batch_data, _ in dataloader:
                batch_data = batch_data.to(self.device)

                with torch.no_grad():
                    z_anchor = anchor(batch_data)

                z_adapter = self.adapter(batch_data)
                loss = self.distillation_loss(z_adapter, z_anchor)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            total_loss += epoch_loss

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def compute_statistics(
        self,
        anchor: nn.Module,
        dataloader: DataLoader,
    ) -> ClientPayload:
        """
        Phase 2: Project local data through Anchor / Adapter and compute empirical statistics (Eq. 4).
        """
        anchor.eval()
        if self.adapter is not None:
            self.adapter.eval()

        class_vectors: Dict[int, list] = {}

        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(self.device)

            # Project into semantic space
            if self.adapter is not None:
                z = self.adapter(batch_data)
            else:
                z = anchor(batch_data)

            for i in range(len(batch_labels)):
                label = batch_labels[i].item()
                if label not in class_vectors:
                    class_vectors[label] = []
                class_vectors[label].append(z[i].cpu())

        # Compute per-class statistics (Eq. 4)
        class_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor, int]] = {}
        total_samples = 0

        for c, vectors in class_vectors.items():
            z_c = torch.stack(vectors)  # (n_c, d)
            n_c = z_c.size(0)
            total_samples += n_c

            if n_c < 2:
                mean_c = z_c.mean(dim=0)
                cov_c = torch.eye(z_c.size(1)) * 0.01
            else:
                # Empirical mean (Eq. 4, line 1)
                mean_c = z_c.mean(dim=0)  # (d,)

                # Empirical covariance (Eq. 4, line 2)
                centered = z_c - mean_c.unsqueeze(0)  # (n_c, d)
                cov_c = (centered.T @ centered) / n_c  # (d, d)

                # Small regularization for numerical PSD stability
                cov_c += torch.eye(cov_c.size(0)) * 1e-4

            class_stats[c] = (mean_c, cov_c, n_c)

        # Differential Privacy extension (Section IV-F)
        if self.privacy_config.enable_dp:
            class_stats = apply_dp_to_statistics(
                class_stats, self.privacy_config
            )

        return ClientPayload(
            client_id=self.client_id,
            class_stats=class_stats,
            total_samples=total_samples,
        )

    def execute_round(
        self,
        anchor: nn.Module,
        dataloader: DataLoader,
    ) -> Tuple[ClientPayload, float]:
        """
        Execute full client round: statistical abstraction.
        """
        dist_loss = 0.0
        if self.adapter is not None:
            dist_loss = self.distill_adapter(anchor, dataloader)

        payload = self.compute_statistics(anchor, dataloader)
        return payload, dist_loss

    def get_payload_size_bytes(self, payload: ClientPayload) -> int:
        """
        Communication cost in bytes.
        For each class: d floats (mean) + d(d+1)/2 floats (upper-tri covariance) + 1 int (count).
        """
        total_bytes = 0
        for c, (mean_c, cov_c, n_c) in payload.class_stats.items():
            d = mean_c.numel()
            total_bytes += d * 4
            total_bytes += (d * (d + 1) // 2) * 4
            total_bytes += 4
        return total_bytes
