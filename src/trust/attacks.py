"""
Byzantine Attack Simulations
==============================
Implements malicious client attacks to test the robustness of FLGSS's
Mahalanobis-based filtering mechanism (Section IV-E, Section VI-B).

Attack types:
  1. Random Noise Attack: Replace statistics with random Gaussian noise.
  2. Label Flipping Attack: Swap class labels before computing statistics.
  3. Scaling Attack: Multiply means by a large factor to shift the GMM.
  4. Gradient-Mimicking Attack: Adaptive attack that crafts statistics close
     to the boundary of the rejection threshold.

Reference: Section VI-B (Byzantine Resilience Evaluation).
"""

from typing import Dict, Tuple
import torch
import numpy as np

from client.flgss_client import ClientPayload


def inject_random_noise_attack(
    payload: ClientPayload,
    noise_scale: float = 10.0,
    seed: int = None,
) -> ClientPayload:
    """
    Random noise attack: Replace all per-class statistics with random noise.

    The attacker sends completely fabricated μ and Σ with large-magnitude
    noise, attempting to corrupt the community GMM.

    Args:
        payload:     Original legitimate payload.
        noise_scale: Magnitude multiplier for the random noise.
        seed:        Random seed for reproducibility.

    Returns:
        Poisoned ClientPayload.
    """
    rng = np.random.default_rng(seed)
    poisoned_stats = {}

    for c, (mean_c, cov_c, n_c) in payload.class_stats.items():
        d = mean_c.size(0)
        # Generate random mean far from the true distribution
        noisy_mean = torch.randn(d) * noise_scale

        # Generate random covariance (ensure PSD)
        random_mat = torch.randn(d, d) * noise_scale
        noisy_cov = random_mat @ random_mat.T / d + torch.eye(d) * 0.01

        poisoned_stats[c] = (noisy_mean, noisy_cov, n_c)

    return ClientPayload(
        client_id=payload.client_id,
        class_stats=poisoned_stats,
        total_samples=payload.total_samples,
    )


def inject_label_flip_attack(
    payload: ClientPayload,
    num_classes: int = 10,
) -> ClientPayload:
    """
    Label flipping attack: Cyclically permute class labels.

    Maps class c to class (c + 1) mod num_classes, causing the
    community GMM to mix up class distributions.

    Args:
        payload:     Original legitimate payload.
        num_classes: Total number of classes.

    Returns:
        Poisoned ClientPayload with shuffled class labels.
    """
    flipped_stats = {}

    for c, stats in payload.class_stats.items():
        # Map class c → (c + 1) mod num_classes
        new_label = (c + 1) % num_classes
        flipped_stats[new_label] = stats

    return ClientPayload(
        client_id=payload.client_id,
        class_stats=flipped_stats,
        total_samples=payload.total_samples,
    )


def inject_scaling_attack(
    payload: ClientPayload,
    scale_factor: float = 100.0,
) -> ClientPayload:
    """
    Scaling attack: Multiply all means by a large factor.

    Shifts the class distributions far from the true values,
    attempting to drag the community mean toward the attacker's position.

    Args:
        payload:      Original legitimate payload.
        scale_factor: Multiplier for the mean vectors.

    Returns:
        Poisoned ClientPayload.
    """
    poisoned_stats = {}

    for c, (mean_c, cov_c, n_c) in payload.class_stats.items():
        poisoned_stats[c] = (mean_c * scale_factor, cov_c, n_c)

    return ClientPayload(
        client_id=payload.client_id,
        class_stats=poisoned_stats,
        total_samples=payload.total_samples,
    )


def inject_attack(
    payload: ClientPayload,
    attack_type: str,
    num_classes: int = 10,
    noise_scale: float = 10.0,
    scale_factor: float = 100.0,
    seed: int = None,
) -> ClientPayload:
    """
    Unified attack injection interface.

    Args:
        payload:      Original legitimate payload.
        attack_type:  "random", "label_flip", or "scaling".
        num_classes:  Number of classes (for label flipping).
        noise_scale:  Noise magnitude (for random attack).
        scale_factor: Scale factor (for scaling attack).
        seed:         Random seed.

    Returns:
        Poisoned ClientPayload.
    """
    if attack_type == "random":
        return inject_random_noise_attack(payload, noise_scale, seed)
    elif attack_type == "label_flip":
        return inject_label_flip_attack(payload, num_classes)
    elif attack_type == "scaling":
        return inject_scaling_attack(payload, scale_factor)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
