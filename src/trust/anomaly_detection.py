"""
Unsupervised Anomaly Detection via Global GMM
===============================================
Uses the global generative model P_Global(z) for trustworthy analytics.

Anomaly score: Score(x) = -log P_Global(z), where z = A_φ(x).
High scores indicate out-of-distribution or anomalous data.

Evaluation: ROC curve and AUC on held-out anomalous classes.

Reference: Section IV-D (Trustworthy Analytics), Section VI-C.
"""

from typing import Tuple

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from server.cloud_server import GlobalGMM


def compute_anomaly_scores(
    global_gmm: GlobalGMM,
    latent_vectors: torch.Tensor,
) -> torch.Tensor:
    """
    Compute anomaly scores for a batch of latent vectors.

    Score(x) = -log P_Global(z)

    Args:
        global_gmm:     The trained global GMM.
        latent_vectors: Tensor of shape (N, d) — latent embeddings.

    Returns:
        scores: Tensor of shape (N,) — anomaly scores (higher = more anomalous).
    """
    log_marginal = global_gmm.log_marginal_likelihood(latent_vectors)
    scores = -log_marginal  # High score = low likelihood = anomaly
    return scores


def evaluate_anomaly_detection(
    global_gmm: GlobalGMM,
    normal_vectors: torch.Tensor,
    anomaly_vectors: torch.Tensor,
) -> dict:
    """
    Evaluate anomaly detection performance using AUC-ROC.

    Normal samples should have low anomaly scores (high likelihood).
    Anomalous samples should have high anomaly scores (low likelihood).

    The paper reports AUC ≈ 0.98 on CIFAR-10 and 0.99 on Intel Berkeley Lab.

    Args:
        global_gmm:      Trained global GMM.
        normal_vectors:  Latent embeddings of normal samples (N_normal, d).
        anomaly_vectors: Latent embeddings of anomalous samples (N_anomaly, d).

    Returns:
        dict with:
            - auc: Area Under the ROC Curve.
            - fpr: False positive rates for ROC curve.
            - tpr: True positive rates for ROC curve.
            - thresholds: Decision thresholds.
    """
    # Compute anomaly scores
    normal_scores = compute_anomaly_scores(global_gmm, normal_vectors)
    anomaly_scores = compute_anomaly_scores(global_gmm, anomaly_vectors)

    # Combine scores and labels
    all_scores = torch.cat([normal_scores, anomaly_scores]).numpy()
    all_labels = np.concatenate([
        np.zeros(len(normal_scores)),   # 0 = normal
        np.ones(len(anomaly_scores)),   # 1 = anomaly
    ])

    # Handle edge cases (NaN, Inf)
    valid_mask = np.isfinite(all_scores)
    all_scores = all_scores[valid_mask]
    all_labels = all_labels[valid_mask]

    if len(np.unique(all_labels)) < 2:
        return {"auc": 0.0, "fpr": [], "tpr": [], "thresholds": []}

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    auc = roc_auc_score(all_labels, all_scores)

    return {
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "normal_scores_mean": float(normal_scores.mean()),
        "anomaly_scores_mean": float(anomaly_scores.mean()),
    }


def find_nearest_normal(
    global_gmm: GlobalGMM,
    z_anomaly: torch.Tensor,
    num_samples: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Explainable AI via counterfactual reasoning.

    Given an anomalous embedding z_anomaly, find the closest "normal" point
    on the high-probability manifold of P_Global(z).

    The semantic change Δz = z* - z_anomaly explains what would need to
    change for the event to be considered normal.

    Reference: Section IV-D (Explainable AI paragraph).

    Args:
        global_gmm:  The global GMM.
        z_anomaly:   Anomalous latent vector (d,).
        num_samples: Number of candidate normal points to sample.

    Returns:
        z_nearest: Closest high-likelihood point (d,).
        delta_z:   Semantic change vector (d,).
    """
    # Sample from each component and find the nearest high-likelihood point
    candidates = []

    for c in range(global_gmm.num_classes):
        for comp in global_gmm.class_components[c]:
            # Sample from this Gaussian component
            try:
                L = torch.linalg.cholesky(comp.covariance)
                n_per_comp = max(num_samples // (
                    sum(len(global_gmm.class_components[cc])
                        for cc in range(global_gmm.num_classes)) or 1
                ), 10)
                noise = torch.randn(n_per_comp, comp.mean.size(0))
                samples = comp.mean.unsqueeze(0) + (noise @ L.T)
                candidates.append(samples)
            except RuntimeError:
                continue

    if not candidates:
        return z_anomaly.clone(), torch.zeros_like(z_anomaly)

    all_candidates = torch.cat(candidates, dim=0)  # (M, d)

    # Find closest candidate by Euclidean distance
    distances = torch.norm(all_candidates - z_anomaly.unsqueeze(0), dim=1)
    nearest_idx = distances.argmin()

    z_nearest = all_candidates[nearest_idx]
    delta_z = z_nearest - z_anomaly

    return z_nearest, delta_z
