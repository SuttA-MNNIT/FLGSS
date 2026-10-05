import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

def apply_ledoit_wolf_shrinkage(cov: torch.Tensor, gamma: float = 0.1, reg_eps: float = 1e-5) -> torch.Tensor:
    """
    Applies Ledoit-Wolf shrinkage to covariance matrix:
    Sigma_tilde = (1 - gamma) * Sigma + gamma * (Tr(Sigma) / d) * I
    Adds reg_eps * I to guarantee strict positive definiteness.
    """
    d = cov.shape[-1]
    eye = torch.eye(d, device=cov.device, dtype=cov.dtype)
    trace_val = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    target = (trace_val / d).unsqueeze(-1) * eye
    shrunk = (1.0 - gamma) * cov + gamma * target + eye * reg_eps
    return shrunk

def compute_geometric_median(points: torch.Tensor, max_iter: int = 50, tol: float = 1e-5) -> torch.Tensor:
    """
    Computes Geometric Median using Weiszfeld's algorithm for round t=1 bootstrapping.
    points: [N, D]
    Returns median: [D]
    """
    if points.shape[0] == 1:
        return points[0]
    
    # Initialize with component-wise mean
    median = points.mean(dim=0)
    for _ in range(max_iter):
        distances = torch.norm(points - median, dim=1, keepdim=True) # [N, 1]
        distances = torch.clamp(distances, min=1e-6)
        weights = 1.0 / distances
        weights_sum = weights.sum()
        new_median = (points * weights).sum(dim=0) / weights_sum
        if torch.norm(new_median - median) < tol:
            break
        median = new_median
    return median

def compute_mahalanobis_anomaly_score(client_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor, int]],
                                      server_means: Dict[int, torch.Tensor],
                                      server_covs: Dict[int, torch.Tensor],
                                      gamma: float = 0.1) -> float:
    """
    Computes normalized Mahalanobis anomaly score S_k across all present classes (Eq. 5):
    S_k = sqrt( total_dist_sq / (d * num_matched_classes) )
    - Honest clients under natural Non-IID variance score ~ 1.0 (0.6 - 1.5).
    - Byzantine malicious poisoning shifts score > 3.0.
    """
    total_dist_sq = 0.0
    num_matched_classes = 0
    d = 0

    for c, (c_mean, _, _) in client_stats.items():
        if c in server_means and c in server_covs:
            s_mean = server_means[c].to(c_mean.device, dtype=c_mean.dtype)
            s_cov = server_covs[c].to(c_mean.device, dtype=c_mean.dtype)
            d = c_mean.shape[-1]

            diff = (c_mean - s_mean).unsqueeze(-1) # [D, 1]
            shrunk_cov = apply_ledoit_wolf_shrinkage(s_cov, gamma=gamma)

            try:
                L = torch.linalg.cholesky(shrunk_cov)
                y = torch.linalg.solve_triangular(L, diff, upper=False)
                dist_sq = torch.sum(y * y).item()
            except RuntimeError:
                inv_cov = torch.linalg.pinv(shrunk_cov)
                dist_sq = (diff.transpose(0, 1) @ inv_cov @ diff).item()

            total_dist_sq += max(0.0, dist_sq)
            num_matched_classes += 1

    if num_matched_classes == 0 or d == 0:
        return 0.0

    return float(np.sqrt(total_dist_sq / (d * num_matched_classes)))

def inject_byzantine_attack(client_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor, int]],
                            attack_type: str = "gaussian_shift",
                            noise_scale: float = 5.0) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Simulates malicious Byzantine poisoning attacks:
    - 'gaussian_shift': shifts class means by large adversarial direction to corrupt GMM centers.
    - 'noise': injects high-variance noise into means and covariances.
    - 'label_flip': rotates/permutates the class labels.
    """
    if attack_type == "none":
        return client_stats

    poisoned = {}
    classes = list(client_stats.keys())

    if attack_type == "label_flip":
        # Permute classes: 0->1, 1->2, ..., K->0
        for i, c in enumerate(classes):
            next_c = classes[(i + 1) % len(classes)]
            poisoned[next_c] = client_stats[c]
        return poisoned

    for c, (mean, cov, count) in client_stats.items():
        p_mean = mean.clone()
        p_cov = cov.clone()

        if attack_type == "gaussian_shift":
            adversarial_shift = torch.randn_like(mean) * noise_scale * 3.0
            p_mean = p_mean + adversarial_shift
        elif attack_type == "noise":
            noise = torch.randn_like(mean) * noise_scale
            p_mean = p_mean + noise
            p_cov = p_cov + torch.eye(cov.shape[-1], device=cov.device) * noise_scale

        poisoned[c] = (p_mean, p_cov, count)

    return poisoned
