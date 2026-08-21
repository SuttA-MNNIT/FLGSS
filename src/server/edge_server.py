"""
Edge Server — Robust Community-Level Aggregation (Phase 3, Part 1)
===================================================================
Implements the edge server from Algorithm 1, Step B:

1. Receives statistical payloads from all participating clients in its community.
2. Validates each update via Mahalanobis-based anomaly scoring (Eq. 6).
3. Rejects malicious/anomalous updates (S_k > τ).
4. Constructs a Community GMM from valid updates (Eq. 5).

The community GMM is periodically transmitted to the Cloud Server for
global synthesis.

Reference: Section IV-D (Hierarchical Aggregation), Section IV-E (Robust Agg).
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import torch
import numpy as np

from config import RobustnessConfig
from client.flgss_client import ClientPayload
from trust.robust_aggregation import (
    compute_anomaly_score,
    compute_geometric_median,
    ledoit_wolf_shrinkage,
)


@dataclass
class GMMComponent:
    """A single component of a Gaussian Mixture Model."""
    weight: float           # π_{k,c}: mixing weight
    mean: torch.Tensor      # μ_{k,c}: mean vector (d,)
    covariance: torch.Tensor  # Σ_{k,c}: covariance matrix (d, d)
    sample_count: int       # |Z_{k,c}|: number of samples
    client_id: int          # Source client identifier
    L: Optional[torch.Tensor] = None        # Precomputed Cholesky factor
    log_det: Optional[torch.Tensor] = None  # Precomputed log determinant



@dataclass
class CommunityGMM:
    """
    Community-level Gaussian Mixture Model for a single class.

    P_j^C(z|c) = Σ_{k∈C_j} π_{k,c} N(z | μ_{k,c}, Σ_{k,c})
    """
    class_label: int
    components: List[GMMComponent] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return sum(comp.sample_count for comp in self.components)

    @property
    def community_mean(self) -> torch.Tensor:
        """Weighted mean of all components."""
        if not self.components:
            return None
        total = sum(c.sample_count for c in self.components)
        weighted_sum = sum(c.sample_count * c.mean for c in self.components)
        return weighted_sum / max(total, 1)

    @property
    def community_covariance(self) -> torch.Tensor:
        """Weighted covariance of all components (for filtering)."""
        if not self.components:
            return None
        total = sum(c.sample_count for c in self.components)
        mu_c = self.community_mean
        weighted_cov = torch.zeros_like(self.components[0].covariance)
        for comp in self.components:
            w = comp.sample_count / max(total, 1)
            # Total variance = within-group variance + between-group variance
            delta = comp.mean - mu_c
            weighted_cov += w * (comp.covariance + delta.unsqueeze(1) @ delta.unsqueeze(0))
        return weighted_cov


class EdgeServer:
    """
    Edge Server managing a community of IoT devices.

    Responsibilities:
    - Receive and validate client statistical payloads.
    - Perform robust aggregation (Mahalanobis filtering).
    - Construct community-level GMM for each class.
    - Periodically upload community models to the cloud.
    """

    def __init__(
        self,
        server_id: int,
        community_client_ids: List[int],
        robustness_config: RobustnessConfig,
        num_classes: int,
        latent_dim: int,
    ):
        self.server_id = server_id
        self.community_client_ids = community_client_ids
        self.robustness_config = robustness_config
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Current community GMM (one per class)
        self.community_gmms: Dict[int, CommunityGMM] = {}

        # Persistent client state table across rounds
        self.client_payloads: Dict[int, ClientPayload] = {}

        # History tracking
        self.round_count = 0
        self.rejected_clients: List[int] = []

    def aggregate_round(
        self,
        payloads: List[ClientPayload],
    ) -> Dict[int, CommunityGMM]:
        """
        Process a round of client payloads:
        1. Filter malicious updates (robust aggregation).
        2. Update persistent community client registry.
        3. Build community GMM from all known valid client updates.

        This implements "Step B: Edge Server Aggregation" from Algorithm 1.

        Args:
            payloads: List of ClientPayload from participating clients.

        Returns:
            community_gmms: Dict mapping class_label → CommunityGMM.
        """
        self.round_count += 1
        self.rejected_clients = []

        # ── Step 1: Robust Filtering ──────────────────────────────────────
        valid_payloads = self._filter_payloads(payloads)

        # Update persistent community client registry with latest valid updates
        for payload in valid_payloads:
            self.client_payloads[payload.client_id] = payload

        # ── Step 2: Build Community GMMs from all known clients in community ─
        new_gmms: Dict[int, CommunityGMM] = {}

        for c in range(self.num_classes):
            gmm = CommunityGMM(class_label=c)

            # Collect all valid (mean, cov, count) for this class across community
            total_count_c = 0
            class_data = []

            for payload in self.client_payloads.values():
                if c in payload.class_stats:
                    mean_c, cov_c, n_c = payload.class_stats[c]
                    class_data.append((payload.client_id, mean_c, cov_c, n_c))
                    total_count_c += n_c

            # Build GMM components with normalized weights (Eq. 5)
            for client_id, mean_c, cov_c, n_c in class_data:
                pi_kc = n_c / max(total_count_c, 1)  # π_{k,c}
                gmm.components.append(GMMComponent(
                    weight=pi_kc,
                    mean=mean_c,
                    covariance=cov_c,
                    sample_count=n_c,
                    client_id=client_id,
                ))

            new_gmms[c] = gmm

        # Update stored community model
        self.community_gmms = new_gmms
        return new_gmms

    def _filter_payloads(
        self,
        payloads: List[ClientPayload],
    ) -> List[ClientPayload]:
        """
        Apply robust filtering to reject anomalous/malicious client updates.

        Round 1 (Bootstrap): Use geometric median + Euclidean distance.
        Round ≥ 2: Use Mahalanobis distance against current community model.

        Reference: Section IV-E, Eq. 6.
        """
        if not self.robustness_config.enable_robust_agg:
            return payloads

        if len(payloads) <= 1:
            return payloads

        valid_payloads = []

        if self.round_count == 1:
            # ── Bootstrap: Geometric Median Filtering ─────────────────────
            valid_payloads = self._bootstrap_filter(payloads)
        else:
            # ── Standard: Mahalanobis Distance Filtering ──────────────────
            for payload in payloads:
                score = self._compute_client_score(payload)

                # Compute threshold τ from χ² distribution
                from scipy.stats import chi2
                dof = self.num_classes * self.latent_dim
                tau = chi2.ppf(
                    self.robustness_config.tau_percentile, df=dof
                )
                tau = np.sqrt(tau)  # We use sqrt in Eq. 6

                if score <= tau:
                    valid_payloads.append(payload)
                else:
                    self.rejected_clients.append(payload.client_id)

        return valid_payloads

    def _bootstrap_filter(
        self,
        payloads: List[ClientPayload],
    ) -> List[ClientPayload]:
        """
        Round-1 bootstrap filtering using geometric median of per-class means.

        Computes geometric median of all μ_{k,c} for each class, then
        filters clients based on aggregate Euclidean distance from medians.
        """
        # Compute geometric median per class
        class_medians: Dict[int, torch.Tensor] = {}

        for c in range(self.num_classes):
            means_c = []
            for p in payloads:
                if c in p.class_stats:
                    means_c.append(p.class_stats[c][0])
            if means_c:
                stacked = torch.stack(means_c)
                class_medians[c] = compute_geometric_median(
                    stacked,
                    max_iters=self.robustness_config.geometric_median_iters,
                )

        # Score each client by aggregate distance from class medians
        scores = []
        for p in payloads:
            total_dist_sq = 0.0
            for c, median_c in class_medians.items():
                if c in p.class_stats:
                    diff = p.class_stats[c][0] - median_c
                    total_dist_sq += (diff ** 2).sum().item()
            scores.append((p, np.sqrt(total_dist_sq)))

        # Use median + k*MAD as threshold
        score_vals = [s for _, s in scores]
        median_score = np.median(score_vals)
        mad = np.median(np.abs(np.array(score_vals) - median_score))
        threshold = median_score + 3.0 * max(mad, 1e-6)

        valid = []
        for payload, score in scores:
            if score <= threshold:
                valid.append(payload)
            else:
                self.rejected_clients.append(payload.client_id)

        return valid

    def _compute_client_score(self, payload: ClientPayload) -> float:
        """
        Compute the anomaly score S_k for a client's update (Eq. 6).

        S_k = sqrt(Σ_c (μ_{k,c} - μ_{j,c}^C)ᵀ Σ̃⁻¹_{j,c} (μ_{k,c} - μ_{j,c}^C))

        Uses Ledoit-Wolf shrinkage for numerical stability.
        """
        total_sq = 0.0

        for c in range(self.num_classes):
            if c not in payload.class_stats:
                continue
            if c not in self.community_gmms:
                continue

            gmm_c = self.community_gmms[c]
            if gmm_c.community_mean is None:
                continue

            mu_k = payload.class_stats[c][0]
            mu_comm = gmm_c.community_mean
            cov_comm = gmm_c.community_covariance

            # Apply Ledoit-Wolf shrinkage
            cov_shrunk = ledoit_wolf_shrinkage(
                cov_comm,
                gamma=self.robustness_config.shrinkage_gamma,
            )

            # Compute class-conditional Mahalanobis distance
            score_c = compute_anomaly_score(mu_k, mu_comm, cov_shrunk)
            total_sq += score_c ** 2

        return np.sqrt(total_sq)

    def get_community_model(self) -> Dict[int, CommunityGMM]:
        """Return the current community GMMs for upload to cloud."""
        return self.community_gmms

    def get_stats(self) -> dict:
        """Return round statistics for logging."""
        total_components = sum(
            len(gmm.components) for gmm in self.community_gmms.values()
        )
        return {
            "server_id": self.server_id,
            "round": self.round_count,
            "total_components": total_components,
            "rejected_clients": self.rejected_clients.copy(),
        }
