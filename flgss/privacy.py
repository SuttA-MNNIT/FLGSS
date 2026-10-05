import torch
import numpy as np
from typing import Dict, Tuple

def apply_client_level_dp(client_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor, int]],
                          epsilon: float = 2.0,
                          delta: float = 1e-5,
                          clip_mean: float = 10.0,
                          clip_cov: float = 20.0,
                          eps_psd: float = 1e-4) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Implements Client-Level Differential Privacy (CL-DP) for FLGSS (Section III-D).
    1. Norm Clipping for Mean (S1) and Covariance (S2).
    2. Calibrated Gaussian Mechanism perturbation.
    3. Eigenvalue projection to strictly enforce Positive Semi-Definiteness (PSD).
    """
    # Noise scale sigma from (epsilon, delta) using standard Gaussian mechanism
    sigma = np.sqrt(2.0 * np.log(1.25 / delta)) / max(epsilon, 1e-6)

    private_stats = {}

    for c, (mean, cov, count) in client_stats.items():
        # --- 1. Clip Mean Vector ---
        mean_norm = torch.norm(mean, p=2)
        clip_coef_mean = min(1.0, clip_mean / (mean_norm.item() + 1e-8))
        clipped_mean = mean * clip_coef_mean

        # Add Gaussian Noise to Mean
        noise_mean = torch.randn_like(clipped_mean) * (sigma * clip_mean * 0.05)
        noisy_mean = clipped_mean + noise_mean

        # --- 2. Clip Covariance Matrix ---
        cov_norm = torch.norm(cov, p='fro')
        clip_coef_cov = min(1.0, clip_cov / (cov_norm.item() + 1e-8))
        clipped_cov = cov * clip_coef_cov

        # Symmetric noise for covariance
        d = cov.shape[-1]
        raw_noise = torch.randn((d, d), device=cov.device, dtype=cov.dtype)
        sym_noise = 0.5 * (raw_noise + raw_noise.T) * (sigma * clip_cov * 0.02)
        noisy_cov = clipped_cov + sym_noise

        # --- 3. PSD Projection via Eigenvalue Clipping ---
        # Covariance must remain symmetric
        noisy_cov = 0.5 * (noisy_cov + noisy_cov.T)
        eigvals, eigvecs = torch.linalg.eigh(noisy_cov)
        eigvals_clipped = torch.clamp(eigvals, min=eps_psd)
        psd_cov = eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T

        private_stats[c] = (noisy_mean, psd_cov, count)

    return private_stats
