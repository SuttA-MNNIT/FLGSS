"""
Robust Aggregation Utilities
==============================
Implements the Mahalanobis-based anomaly scoring and geometric median
bootstrap for robust filtering of malicious client updates.

Key functions:
  - Ledoit-Wolf shrinkage estimator for well-conditioned covariance inversion.
  - Mahalanobis distance computation (Eq. 6).
  - Geometric median via Weiszfeld's algorithm (round-1 bootstrap).

Reference: Section IV-E of the paper.
"""

import torch
import numpy as np


def ledoit_wolf_shrinkage(
    cov: torch.Tensor,
    gamma: float = 0.1,
) -> torch.Tensor:
    """
    Apply Ledoit-Wolf shrinkage to a covariance matrix for numerical stability.

    Σ̃ = (1 - γ)Σ + γ · (Tr(Σ)/d) · I

    This regularizes the covariance toward a scaled identity, ensuring
    the matrix is well-conditioned and invertible.

    Args:
        cov:   Raw covariance matrix (d, d).
        gamma: Shrinkage coefficient γ ∈ [0, 1].

    Returns:
        Shrunk covariance matrix Σ̃.
    """
    d = cov.size(0)
    trace_over_d = torch.trace(cov) / d
    identity = torch.eye(d, dtype=cov.dtype, device=cov.device)

    cov_shrunk = (1 - gamma) * cov + gamma * trace_over_d * identity

    return cov_shrunk


def compute_anomaly_score(
    mu_client: torch.Tensor,
    mu_community: torch.Tensor,
    cov_community: torch.Tensor,
) -> float:
    """
    Compute the Mahalanobis distance between a client's mean and the
    community mean, using the (shrunk) community covariance.

    d_M = sqrt((μ_k - μ_C)ᵀ Σ̃⁻¹ (μ_k - μ_C))

    This is one term of the full anomaly score S_k in Eq. 6.

    Args:
        mu_client:    Client mean vector μ_{k,c} of shape (d,).
        mu_community: Community mean vector μ_{j,c}^C of shape (d,).
        cov_community: Shrunk community covariance Σ̃_{j,c}^C of shape (d, d).

    Returns:
        Mahalanobis distance (scalar).
    """
    diff = mu_client - mu_community  # (d,)

    # Solve via Cholesky for numerical stability
    try:
        L = torch.linalg.cholesky(cov_community)
        v = torch.linalg.solve_triangular(L, diff.unsqueeze(1), upper=False)
        mahal_sq = (v ** 2).sum().item()
    except RuntimeError:
        # Fallback: direct inversion with regularization
        d = cov_community.size(0)
        cov_reg = cov_community + torch.eye(d) * 1e-4
        cov_inv = torch.linalg.inv(cov_reg)
        mahal_sq = (diff @ cov_inv @ diff).item()

    return np.sqrt(max(mahal_sq, 0.0))


def compute_geometric_median(
    points: torch.Tensor,
    max_iters: int = 50,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the geometric median of a set of points using Weiszfeld's algorithm.

    The geometric median minimizes the sum of Euclidean distances to all points:
        gm = argmin_y Σ_i ||y - x_i||₂

    Unlike the arithmetic mean, it is robust to outliers (breakdown point ≈ 50%).
    Used for bootstrapping the robust filter in round 1.

    Args:
        points:    Tensor of shape (N, d) — the points.
        max_iters: Maximum number of Weiszfeld iterations.
        tol:       Convergence tolerance.

    Returns:
        Geometric median of shape (d,).
    """
    # Initialize with the component-wise median
    y = torch.median(points, dim=0).values.clone()

    for _ in range(max_iters):
        # Compute distances from current estimate to all points
        diffs = points - y.unsqueeze(0)       # (N, d)
        distances = torch.norm(diffs, dim=1)  # (N,)

        # Avoid division by zero
        distances = torch.clamp(distances, min=1e-10)

        # Weiszfeld update weights: w_i = 1 / ||y - x_i||
        weights = 1.0 / distances  # (N,)

        # Weighted average
        y_new = (weights.unsqueeze(1) * points).sum(dim=0) / weights.sum()

        # Check convergence
        shift = torch.norm(y_new - y).item()
        y = y_new

        if shift < tol:
            break

    return y


def compute_chi2_threshold(
    num_classes: int,
    latent_dim: int,
    percentile: float = 0.999,
) -> float:
    """
    Compute the Mahalanobis distance threshold τ based on the χ² distribution.

    Under the null hypothesis (benign client), the squared Mahalanobis distance
    follows a χ² distribution with |Y|·d degrees of freedom.

    Args:
        num_classes: |Y| — number of classes.
        latent_dim:  d — dimension of the semantic space.
        percentile:  Desired percentile (e.g., 0.999 for 99.9%).

    Returns:
        τ: The rejection threshold.
    """
    from scipy.stats import chi2
    dof = num_classes * latent_dim
    tau_sq = chi2.ppf(percentile, df=dof)
    return np.sqrt(tau_sq)
