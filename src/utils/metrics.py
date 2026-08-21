"""
Evaluation Metrics
===================
Computes all metrics from the paper's evaluation framework (Section V-C):

1. Model Performance: Top-1 accuracy on global test set.
2. Communication Efficiency: Total bytes transmitted per round and cumulative.
3. Trustworthiness: AUC for anomaly detection.
4. Convergence tracking: per-round accuracy for plotting.

All results support 95% confidence intervals over multiple runs.
"""

from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute Top-1 accuracy.

    Args:
        predictions: (N,) predicted class labels.
        labels:      (N,) ground truth labels.

    Returns:
        accuracy: Float in [0, 1].
    """
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / max(total, 1)


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> tuple:
    """
    Compute mean and 95% confidence interval.

    Args:
        values:     List of metric values across runs.
        confidence: Confidence level (default 0.95).

    Returns:
        (mean, ci_low, ci_high)
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0

    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0

    from scipy.stats import t
    alpha = 1 - confidence
    t_val = t.ppf(1 - alpha / 2, df=max(n - 1, 1))
    margin = t_val * std / np.sqrt(n)

    return mean, mean - margin, mean + margin


class MetricsTracker:
    """
    Tracks all metrics across rounds and runs for a single experiment.
    """

    def __init__(self, method_name: str):
        self.method_name = method_name

        # Per-round tracking (current run)
        self.round_accuracies: List[float] = []
        self.round_comm_costs: List[float] = []  # bytes per round
        self.cumulative_comm: float = 0.0

        # Multi-run tracking
        self.all_run_final_acc: List[float] = []
        self.all_run_histories: List[List[float]] = []
        self.all_run_total_comm: List[float] = []

    def log_round(
        self,
        round_num: int,
        accuracy: float,
        comm_cost_bytes: float,
    ):
        """Log metrics for a single round."""
        self.round_accuracies.append(accuracy)
        self.round_comm_costs.append(comm_cost_bytes)
        self.cumulative_comm += comm_cost_bytes

    def finalize_run(self):
        """Called at end of a run to store history and reset."""
        if self.round_accuracies:
            self.all_run_final_acc.append(self.round_accuracies[-1])
            self.all_run_histories.append(self.round_accuracies.copy())
            self.all_run_total_comm.append(self.cumulative_comm)

        # Reset for next run
        self.round_accuracies = []
        self.round_comm_costs = []
        self.cumulative_comm = 0.0

    def get_summary(self) -> dict:
        """
        Get summary statistics across all runs.

        Returns dict with mean, CI for accuracy and communication.
        """
        acc_mean, acc_lo, acc_hi = compute_confidence_interval(
            self.all_run_final_acc
        )
        comm_mean, comm_lo, comm_hi = compute_confidence_interval(
            self.all_run_total_comm
        )

        return {
            "method": self.method_name,
            "accuracy_mean": acc_mean * 100,
            "accuracy_ci": (acc_lo * 100, acc_hi * 100),
            "comm_cost_mb_mean": comm_mean / (1024 * 1024),
            "comm_cost_mb_ci": (comm_lo / (1024 * 1024), comm_hi / (1024 * 1024)),
            "comm_to_70pct_mb": self.get_comm_to_target(target_acc=0.70),
            "num_runs": len(self.all_run_final_acc),
        }

    def get_comm_to_target(self, target_acc: float = 0.70) -> float:
        """
        Compute total communication cost (MB) required to first reach target accuracy.
        Matches Section VI-B (Figure 2) of the paper.
        """
        if not self.all_run_histories:
            return 0.0
        
        cum_costs = []
        for hist in self.all_run_histories:
            reached_idx = None
            for idx, acc in enumerate(hist):
                if acc >= target_acc:
                    reached_idx = idx
                    break
            if reached_idx is not None:
                # Cumulative cost up to that round
                comm_up_to = sum(self.round_comm_costs[:reached_idx + 1]) if self.round_comm_costs else (self.cumulative_comm / max(len(hist), 1)) * (reached_idx + 1)
                cum_costs.append(comm_up_to / (1024 * 1024))
            else:
                cum_costs.append(self.cumulative_comm / (1024 * 1024))
        return float(np.mean(cum_costs)) if cum_costs else 0.0

    def get_convergence_history(self) -> tuple:
        """
        Get mean convergence curve with confidence intervals.

        Returns:
            rounds: array of round numbers.
            mean_acc: mean accuracy per round.
            ci_low: lower CI per round.
            ci_high: upper CI per round.
        """
        if not self.all_run_histories:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Pad histories to same length
        max_len = max(len(h) for h in self.all_run_histories)
        padded = []
        for h in self.all_run_histories:
            if len(h) < max_len:
                h = h + [h[-1]] * (max_len - len(h))
            padded.append(h)

        arr = np.array(padded)  # (num_runs, num_rounds)
        rounds = np.arange(1, max_len + 1)
        mean_acc = arr.mean(axis=0)
        std_acc = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(max_len)

        from scipy.stats import t
        n = arr.shape[0]
        t_val = t.ppf(0.975, df=max(n - 1, 1))
        margin = t_val * std_acc / np.sqrt(n)

        return rounds, mean_acc, mean_acc - margin, mean_acc + margin


@torch.no_grad()
def evaluate_global_model_flgss(
    cloud_server,
    anchor: torch.nn.Module,
    adapter: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """
    Evaluate FLGSS global model accuracy on the test set.

    Pipeline: x → Anchor(x) or Adapter(x) → z → GMM classify → ŷ

    Args:
        cloud_server: CloudServer with trained global GMM.
        anchor:       Frozen Anchor Model (or adapter for deployment).
        adapter:      Trained adapter (used if anchor is None).
        test_loader:  DataLoader for the test set.
        device:       Device for computation.

    Returns:
        accuracy: Top-1 accuracy on the test set.
    """
    anchor.eval()
    if adapter is not None:
        adapter.eval()

    all_preds = []
    all_labels = []

    for batch_data, batch_labels in test_loader:
        batch_data = batch_data.to(device)

        # Project to latent space using anchor
        z = anchor(batch_data)

        # Classify via global GMM (keeps tensor on device for GPU speedup)
        preds, _ = cloud_server.classify(z)

        all_preds.append(preds.cpu())
        all_labels.append(batch_labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return compute_accuracy(all_preds, all_labels)


@torch.no_grad()
def evaluate_baseline_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """
    Evaluate baseline (FedAvg/FedProx) model accuracy on the test set.

    Args:
        model:       BaselineClassifier with trained head.
        test_loader: DataLoader for the test set.
        device:      Device for computation.

    Returns:
        accuracy: Top-1 accuracy on the test set.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    for batch_data, batch_labels in test_loader:
        batch_data = batch_data.to(device)
        logits = model(batch_data)
        preds = logits.argmax(dim=1).cpu()

        all_preds.append(preds)
        all_labels.append(batch_labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return compute_accuracy(all_preds, all_labels)

