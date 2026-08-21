"""
Client-Level Differential Privacy for FLGSS
=============================================
Implements the DP extension from Section IV-F of the paper:

1. Clipping: L2-norm clipping of μ (threshold S₁) and Σ (threshold S₂).
2. Gaussian Noise: Add calibrated noise N(0, σ²S²I) to clipped statistics.
3. PSD Repair: Eigendecompose noisy Σ and clip negative eigenvalues to ε_PSD.

This provides (ε, δ)-Client-Level Differential Privacy, ensuring the federated
output is statistically indistinguishable whether or not any single client
participated (Theorem 3).

Reference: Section IV-F, Eq. 7-8.
"""

from typing import Dict, Tuple

import torch
import numpy as np

from config import PrivacyConfig


def clip_vector(v: torch.Tensor, max_norm: float) -> torch.Tensor:
    """
    Clip a vector to have L2 norm at most max_norm.

    Args:
        v:        Input vector of shape (d,).
        max_norm: Maximum allowed L2 norm (S₁ or S₂).

    Returns:
        Clipped vector: v * min(1, max_norm / ||v||₂).
    """
    norm = torch.norm(v, p=2)
    if norm > max_norm:
        v = v * (max_norm / norm)
    return v


def clip_matrix(M: torch.Tensor, max_norm: float) -> torch.Tensor:
    """
    Clip the Frobenius norm of a matrix (flattened L2 norm).

    Args:
        M:        Input matrix of shape (d, d).
        max_norm: Maximum allowed Frobenius norm (S₂).

    Returns:
        Clipped matrix.
    """
    norm = torch.norm(M, p="fro")
    if norm > max_norm:
        M = M * (max_norm / norm)
    return M


def add_gaussian_noise(
    tensor: torch.Tensor,
    sensitivity: float,
    epsilon: float,
    delta: float,
) -> torch.Tensor:
    """
    Add calibrated Gaussian noise for (ε, δ)-DP via the Gaussian mechanism.

    The noise standard deviation is:
        σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε

    Args:
        tensor:      Input tensor (mean vector or flattened covariance).
        sensitivity: L2 sensitivity (= clipping norm).
        epsilon:     Privacy budget ε.
        delta:       Privacy parameter δ.

    Returns:
        Noisy tensor: tensor + N(0, σ²I).
    """
    # Compute noise scale (Gaussian mechanism)
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    noise = torch.randn_like(tensor) * sigma
    return tensor + noise


def repair_psd(
    cov: torch.Tensor,
    epsilon_psd: float = 1e-6,
) -> torch.Tensor:
    """
    Ensure a covariance matrix is Positive Semi-Definite (PSD) by
    clipping negative eigenvalues.

    After adding Gaussian noise, the covariance may lose its PSD property.
    We eigendecompose, clip negative eigenvalues to ε_PSD, and reconstruct.

    Args:
        cov:         Possibly non-PSD covariance matrix (d, d).
        epsilon_psd: Minimum eigenvalue (ε_PSD > 0).

    Returns:
        PSD-repaired covariance matrix.
    """
    # Eigendecomposition (symmetric matrix)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Clip negative eigenvalues
    eigenvalues = torch.clamp(eigenvalues, min=epsilon_psd)

    # Reconstruct: Σ_repaired = V diag(λ_clipped) Vᵀ
    cov_repaired = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T

    # Ensure exact symmetry (numerical precision)
    cov_repaired = (cov_repaired + cov_repaired.T) / 2

    return cov_repaired


def apply_dp_to_statistics(
    class_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor, int]],
    config: PrivacyConfig,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Apply the full DP pipeline to a client's class-conditional statistics:
    1. Clip μ and Σ.
    2. Add calibrated Gaussian noise.
    3. Repair Σ to ensure PSD.

    This implements the "Clipping and Perturbing Statistics" paragraph
    from Section IV-F.

    Args:
        class_stats: Dict mapping class → (mean, covariance, count).
        config:      PrivacyConfig with ε, δ, clipping norms.

    Returns:
        Differentially private class statistics.
    """
    if not config.enable_dp:
        return class_stats

    dp_stats = {}

    for c, (mean_c, cov_c, n_c) in class_stats.items():
        # ── Step 1: Clip ──────────────────────────────────────────────────
        mean_clipped = clip_vector(mean_c, config.clip_norm_mean)
        cov_clipped = clip_matrix(cov_c, config.clip_norm_cov)

        # ── Step 2: Add Gaussian Noise ────────────────────────────────────
        mean_noisy = add_gaussian_noise(
            mean_clipped, config.clip_norm_mean,
            config.epsilon, config.delta,
        )
        cov_noisy = add_gaussian_noise(
            cov_clipped, config.clip_norm_cov,
            config.epsilon, config.delta,
        )

        # ── Step 3: PSD Repair ────────────────────────────────────────────
        # Ensure symmetry first
        cov_noisy = (cov_noisy + cov_noisy.T) / 2
        cov_repaired = repair_psd(cov_noisy, config.psd_epsilon)

        dp_stats[c] = (mean_noisy, cov_repaired, n_c)

    return dp_stats
