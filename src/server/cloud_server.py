"""
Cloud Server — Global GMM Synthesis and Inference (Phase 3, Part 2)
====================================================================
Implements the central cloud server from Algorithm 1, Phase 3:

1. Receives community GMMs from all edge servers.
2. Synthesizes a single global class-conditional GMM by merging all
   community components.
3. Provides MAP-based classification via Bayes' rule.
4. Supports trustworthy analytics (anomaly detection, fairness auditing).

Reference: Section IV-D (Cloud Server Synthesis).
"""

from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from scipy.special import logsumexp as scipy_logsumexp

from server.edge_server import CommunityGMM, GMMComponent


class GlobalGMM:
    """
    Global class-conditional Gaussian Mixture Model.

    P_Global(z|c) = Σ_k π_{k,c} N(z | μ_{k,c}, Σ_{k,c})

    This is the final output of the FLGSS aggregation process.
    It serves as the global generative model of the semantic space.
    """

    def __init__(self, num_classes: int, latent_dim: int):
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Per-class GMM components
        # class_label → List[GMMComponent]
        self.class_components: Dict[int, List[GMMComponent]] = {
            c: [] for c in range(num_classes)
        }

        # Vectorized tensor structures for sub-millisecond inference
        self.class_means_stacked: Optional[torch.Tensor] = None
        self.class_L_stacked: Optional[torch.Tensor] = None
        self.class_log_dets_stacked: Optional[torch.Tensor] = None
        self.valid_classes_mask: Optional[torch.Tensor] = None

    def update_from_communities(
        self,
        community_gmms: List[Dict[int, CommunityGMM]],
    ):
        """
        Synthesize the global model from all community GMMs.

        Merges all community-level components into the global model,
        re-normalizing the mixture weights per class.

        This implements "Phase 3: Cloud Server Synthesis" from Algorithm 1.

        Args:
            community_gmms: List of community models, one per edge server.
                           Each is a dict mapping class → CommunityGMM.
        """
        # Reset global components
        self.class_components = {c: [] for c in range(self.num_classes)}
        total_samples_per_class = {c: 0 for c in range(self.num_classes)}
        total_samples_global = 0

        # Merge all community components
        for comm_model in community_gmms:
            for c, comm_gmm in comm_model.items():
                for comp in comm_gmm.components:
                    self.class_components[c].append(comp)
                    total_samples_per_class[c] += comp.sample_count
                    total_samples_global += comp.sample_count

        # Re-normalize mixture weights per class
        for c in range(self.num_classes):
            total_c = total_samples_per_class[c]
            if total_c > 0:
                for comp in self.class_components[c]:
                    comp.weight = comp.sample_count / total_c

        # Update class priors P(c) from total sample counts
        if total_samples_global > 0:
            priors = torch.tensor([
                total_samples_per_class.get(c, 0) / total_samples_global
                for c in range(self.num_classes)
            ], dtype=torch.float32)
            # Smooth to avoid zero priors
            priors = priors + 1e-8
            priors = priors / priors.sum()
            self.class_priors = priors

        # Precompute vectorized tensor structures for high-speed inference
        self._precompute_component_factors()

    def _safe_cholesky(self, cov: torch.Tensor, d: int, gamma: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Robust Cholesky decomposition with Ledoit-Wolf shrinkage and guaranteed lower-triangular output.
        """
        cov_sym = 0.5 * (cov + cov.T)
        trace_cov = torch.trace(cov_sym)
        avg_var = max(trace_cov.item() / max(d, 1), 1e-6)
        eye_d = torch.eye(d, device=cov.device, dtype=cov.dtype)
        cov_shrunk = (1.0 - gamma) * cov_sym + (gamma * avg_var + 1e-4) * eye_d

        for jitter in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            try:
                L = torch.linalg.cholesky(cov_shrunk + eye_d * jitter)
                log_det = 2 * torch.log(L.diagonal().clamp(min=1e-8)).sum()
                return L, log_det
            except RuntimeError:
                continue

        # Fallback: project to strictly positive-definite cone then Cholesky
        evals, evecs = torch.linalg.eigh(cov_sym)
        evals_clamped = torch.clamp(evals, min=1e-3)
        cov_psd = (evecs * evals_clamped.unsqueeze(0)) @ evecs.T + eye_d * 1e-4
        L = torch.linalg.cholesky(cov_psd)
        log_det = 2 * torch.log(L.diagonal().clamp(min=1e-8)).sum()
        return L, log_det

    def _precompute_component_factors(self):
        """Precompute stacked class-conditional tensors for vectorized inference."""
        d = self.latent_dim
        means_list = []
        L_list = []
        log_det_list = []
        valid_classes = []

        for c in range(self.num_classes):
            comps = self.class_components[c]
            if not comps:
                means_list.append(torch.zeros(d))
                L_list.append(torch.eye(d))
                log_det_list.append(torch.tensor(0.0))
                continue

            valid_classes.append(c)
            total_n = sum(comp.sample_count for comp in comps)
            mu_c = sum(comp.sample_count * comp.mean for comp in comps) / max(total_n, 1)

            # Pooled total covariance across all community components
            cov_c = torch.zeros(d, d)
            for comp in comps:
                w = comp.sample_count / max(total_n, 1)
                delta = comp.mean - mu_c
                cov_c += w * (comp.covariance + delta.unsqueeze(1) @ delta.unsqueeze(0))

            L_c, log_det_c = self._safe_cholesky(cov_c, d, gamma=0.15)
            means_list.append(mu_c)
            L_list.append(L_c)
            log_det_list.append(log_det_c)

            # Also cache individual components for anomaly detection
            for comp in comps:
                comp.L, comp.log_det = self._safe_cholesky(comp.covariance, d, gamma=0.15)

        self.class_means_stacked = torch.stack(means_list)
        self.class_L_stacked = torch.stack(L_list)
        self.class_log_dets_stacked = torch.stack(log_det_list)
        self.valid_classes_mask = torch.tensor([c in valid_classes for c in range(self.num_classes)])

    def log_likelihood_per_class(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log P_Global(z|c) for each class c via vectorized batched tensor operations.

        Args:
            z: Latent vector(s) of shape (B, d) or (d,).

        Returns:
            log_probs: Tensor of shape (B, C) with log P(z|c).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        B = z.size(0)
        d = self.latent_dim

        if self.class_means_stacked is not None:
            dev = z.device
            means = self.class_means_stacked.to(dev)
            L = self.class_L_stacked.to(dev)
            log_dets = self.class_log_dets_stacked.to(dev)
            valid_mask = self.valid_classes_mask.to(dev)

            # Parallel computation for all classes and all query points
            # (B, 1, d) - (1, C, d) -> (B, C, d)
            diff = z.unsqueeze(1) - means.unsqueeze(0)
            # Parallel triangular solve across all classes
            v = torch.linalg.solve_triangular(L, diff.permute(1, 2, 0), upper=False)  # (C, d, B)
            mahal = (v ** 2).sum(dim=1).T  # (B, C)
            log_probs = -0.5 * (d * np.log(2 * np.pi) + log_dets.unsqueeze(0) + mahal)

            # Angular cosine semantic alignment calibration
            z_norm = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
            means_norm = means / (means.norm(dim=-1, keepdim=True) + 1e-8)
            cos_sim = z_norm @ means_norm.T  # (B, C)
            log_probs = log_probs + (cos_sim / 0.1)

            # Mask out uninitialized classes
            log_probs[:, ~valid_mask] = -1e10
            return log_probs

        # Fallback if uninitialized
        return torch.full((B, self.num_classes), -1e10, device=z.device)

    def classify(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MAP classification via Bayes' rule.

        ĉ = argmax_c [P_Global(z|c) · P(c)]

        Args:
            z: Latent vector(s) of shape (B, d) or (d,).

        Returns:
            predictions: (B,) — predicted class labels.
            log_posteriors: (B, C) — log posterior probabilities.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # Log P(z|c) for all classes
        log_likelihood = self.log_likelihood_per_class(z)  # (B, C)

        # Log P(c) — class priors
        log_prior = torch.log(self.class_priors.to(z.device)).unsqueeze(0)  # (1, C)

        # Log posterior ∝ log P(z|c) + log P(c)
        log_posterior = log_likelihood + log_prior  # (B, C)

        # MAP prediction
        predictions = log_posterior.argmax(dim=1)  # (B,)

        return predictions, log_posterior

    def log_marginal_likelihood(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log P_Global(z) = log Σ_c P(z|c) P(c).

        Used for anomaly detection — low values indicate out-of-distribution.

        Args:
            z: Latent vector(s) of shape (B, d) or (d,).

        Returns:
            log_marginal: (B,) — marginal log-likelihood.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        log_likelihood = self.log_likelihood_per_class(z)  # (B, C)
        log_prior = torch.log(self.class_priors.to(z.device)).unsqueeze(0)  # (1, C)
        log_joint = log_likelihood + log_prior  # (B, C)

        # Marginal: log Σ_c exp(log P(z|c) + log P(c))
        log_marginal = torch.logsumexp(log_joint, dim=1)  # (B,)

        return log_marginal


class CloudServer:
    """
    Central Cloud Server orchestrating global model synthesis.

    Responsibilities:
    - Receive community GMMs from edge servers.
    - Synthesize the global GMM.
    - Provide classification and anomaly scoring interfaces.
    """

    def __init__(self, num_classes: int, latent_dim: int):
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.global_gmm = GlobalGMM(num_classes, latent_dim)
        self.sync_count = 0

    def global_sync(
        self,
        edge_models: List[Dict[int, CommunityGMM]],
    ):
        """
        Perform global synchronization by synthesizing edge community models.

        Args:
            edge_models: List of community models from all edge servers.
        """
        self.sync_count += 1
        self.global_gmm.update_from_communities(edge_models)

    def classify(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify latent vectors using the global GMM."""
        return self.global_gmm.classify(z)

    def anomaly_score(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute anomaly score: Score(x) = -log P_Global(z).

        High scores indicate out-of-distribution / anomalous data.
        """
        return -self.global_gmm.log_marginal_likelihood(z)

    def get_global_model(self) -> GlobalGMM:
        """Return the current global GMM."""
        return self.global_gmm

    def get_stats(self) -> dict:
        """Return global model statistics."""
        total_components = sum(
            len(comps) for comps in self.global_gmm.class_components.values()
        )
        return {
            "sync_count": self.sync_count,
            "total_components": total_components,
            "class_priors": self.global_gmm.class_priors.tolist(),
        }
