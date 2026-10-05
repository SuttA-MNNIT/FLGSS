import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from .defense import apply_ledoit_wolf_shrinkage, compute_geometric_median, compute_mahalanobis_anomaly_score

class GlobalGMM:
    """
    Hierarchical Gaussian Generative Semantic Space (Section III-B & III-C).
    Maintains class-conditional probability distributions P_Global(z | c).
    Uses Mahalanobis / Gaussian Discriminant Analysis (GDA) in the shared semantic space
    regularized by Ledoit-Wolf covariance shrinkage.
    """
    def __init__(self, num_classes: int, latent_dim: int, device: str = "cuda"):
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.device = device

        # Persistent registry of components per class: class_c -> {client_id: (mean, cov, count)}
        self.client_registry: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor, int]]] = {
            c: {} for c in range(num_classes)
        }

        # Class sample counts P(c)
        self.class_sample_counts = torch.zeros(num_classes, device=device, dtype=torch.float32)

        # Pooled server class summaries
        self.server_means: Dict[int, torch.Tensor] = {}
        self.server_covs: Dict[int, torch.Tensor] = {}
        self.inv_covs: Dict[int, torch.Tensor] = {}
        
        # Global pooled covariance across all classes (Linear Discriminant representation)
        self.pooled_inv_cov: Optional[torch.Tensor] = None
        self.has_model: bool = False

    def is_initialized(self) -> bool:
        return self.has_model

    def update_from_clients(self,
                            accepted_updates: List,
                            gamma: float = 0.15,
                            reg_eps: float = 1e-4):
        """
        Synthesizes client updates into the global class-conditional GMM (Phase 3).
        Preserves client components across rounds via hierarchical GMM pooling.
        """
        if not accepted_updates:
            return

        # 1. Register / refresh updates from active clients
        for item in accepted_updates:
            if isinstance(item, tuple) and len(item) == 2:
                cid, client_dict = item
            else:
                cid, client_dict = id(item), item

            for c, (mean, cov, count) in client_dict.items():
                if count > 0:
                    self.client_registry[c][cid] = (mean.detach().cpu(), cov.detach().cpu(), count)

        # 2. Re-synthesize global class statistics
        total_samples_all = 0.0
        weighted_cov_sum = torch.zeros((self.latent_dim, self.latent_dim), device=self.device, dtype=torch.float32)

        for c in range(self.num_classes):
            reg = self.client_registry[c]
            if not reg:
                continue

            all_items = list(reg.values())
            means = torch.stack([it[0] for it in all_items]).to(self.device)
            raw_covs = torch.stack([it[1] for it in all_items]).to(self.device)
            counts = torch.tensor([it[2] for it in all_items], device=self.device, dtype=torch.float32)
            c_total = counts.sum()
            self.class_sample_counts[c] = c_total
            total_samples_all += c_total.item()

            weights = (counts / c_total).unsqueeze(-1) # [K, 1]

            # Exact pooled mean
            pooled_mean = (means * weights).sum(dim=0) # [D]
            diff = means - pooled_mean.unsqueeze(0) # [K, D]

            # Between-component variance correction
            between_cov = (weights.unsqueeze(-1) * torch.bmm(diff.unsqueeze(2), diff.unsqueeze(1))).sum(dim=0)
            within_cov = (raw_covs * weights.unsqueeze(-1)).sum(dim=0)
            pooled_cov = within_cov + between_cov

            # Ledoit-Wolf Shrinkage Regularization
            shrunk_cov = apply_ledoit_wolf_shrinkage(pooled_cov, gamma=gamma, reg_eps=reg_eps)

            self.server_means[c] = pooled_mean
            self.server_covs[c] = shrunk_cov

            try:
                self.inv_covs[c] = torch.linalg.pinv(shrunk_cov)
            except Exception:
                self.inv_covs[c] = torch.eye(self.latent_dim, device=self.device)

            weighted_cov_sum += shrunk_cov * c_total

        if total_samples_all > 0:
            global_pooled_cov = weighted_cov_sum / total_samples_all
            try:
                self.pooled_inv_cov = torch.linalg.pinv(global_pooled_cov)
            except Exception:
                self.pooled_inv_cov = torch.eye(self.latent_dim, device=self.device)
            self.has_model = True

    @torch.inference_mode()
    def compute_class_log_likelihood(self, z: torch.Tensor, c: int) -> torch.Tensor:
        """
        Computes log P_Global(z | c) in the semantic space.
        Evaluates Gaussian log-density using regularized Mahalanobis distance.
        """
        if c not in self.server_means:
            return torch.full((z.shape[0],), -1e9, device=z.device, dtype=torch.float32)

        mu = self.server_means[c].to(z.device, dtype=torch.float32)
        inv_cov = self.pooled_inv_cov if self.pooled_inv_cov is not None else self.inv_covs.get(c)
        if inv_cov is None:
            inv_cov = torch.eye(self.latent_dim, device=z.device)
        else:
            inv_cov = inv_cov.to(z.device, dtype=torch.float32)

        diff = z.float() - mu.unsqueeze(0) # [B, D]
        # Mahalanobis distance squared: (z - mu)^T Sigma^{-1} (z - mu)
        dist_sq = torch.sum((diff @ inv_cov) * diff, dim=1) # [B]
        
        # log P(z|c) = -0.5 * dist_sq
        return -0.5 * dist_sq

    @torch.inference_mode()
    def predict(self, z: torch.Tensor, use_balanced_prior: bool = True) -> torch.Tensor:
        """
        Inference via Bayes' Rule / Maximum A Posteriori (MAP):
        hat{c} = argmax_c [ log P_Global(z | c) + log P(c) ]
        """
        B = z.shape[0]
        if not self.has_model:
            return torch.zeros(B, device=z.device, dtype=torch.long)

        if use_balanced_prior:
            # Balanced test evaluation (standard in CIFAR-10 / benchmark test sets)
            log_priors = torch.zeros(self.num_classes, device=z.device, dtype=torch.float32)
        else:
            total_samples = max(1.0, self.class_sample_counts.sum().item())
            priors = (self.class_sample_counts + 1.0) / (total_samples + self.num_classes)
            log_priors = torch.log(priors)

        class_log_posteriors = torch.empty((B, self.num_classes), device=z.device, dtype=torch.float32)

        for c in range(self.num_classes):
            log_p_z_given_c = self.compute_class_log_likelihood(z, c)
            class_log_posteriors[:, c] = log_p_z_given_c + log_priors[c]

        return torch.argmax(class_log_posteriors, dim=1)

    @torch.inference_mode()
    def compute_anomaly_scores(self, z: torch.Tensor) -> torch.Tensor:
        """
        Anomaly score for unsupervised OOD / anomaly detection (Section IV-C, Table II):
        Score(z) = - max_c [ log P_Global(z | c) ]
        Lower likelihood under all normal classes -> higher anomaly score.
        """
        B = z.shape[0]
        if not self.has_model:
            return torch.zeros(B, device=z.device, dtype=torch.float32)

        max_log_lik = torch.full((B,), -1e9, device=z.device, dtype=torch.float32)

        for c in self.server_means.keys():
            log_p = self.compute_class_log_likelihood(z, c)
            max_log_lik = torch.maximum(max_log_lik, log_p)

        return -max_log_lik
