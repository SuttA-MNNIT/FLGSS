"""
Visualization Utilities
========================
Produces all plots from the paper's results section:

1. Convergence curves: Test accuracy vs. communication rounds (Fig. 3).
2. Communication cost: Lollipop/bar chart comparison (Fig. 4).
3. Ablation plots: Anchor sensitivity, latent dimension trade-off, privacy-utility.
4. Anomaly detection: ROC curves.

All plots use publication-quality styling matching IEEE Transactions format.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

from utils.metrics import MetricsTracker


# Color palette for methods
METHOD_COLORS = {
    "FLGSS": "#1a9988",
    "FedAvg": "#e74c3c",
    "FedProx": "#9b59b6",
    "SCAFFOLD": "#3498db",
    "MOON": "#f39c12",
    "FedSAM": "#2ecc71",
    "FedKD": "#e67e22",
    "FedClustering": "#1abc9c",
    "PFedKD": "#34495e",
}


def plot_convergence(
    trackers: Dict[str, MetricsTracker],
    title: str = "Test Accuracy vs. Communication Rounds",
    save_path: Optional[str] = None,
):
    """
    Plot convergence curves with 95% confidence intervals.

    Matches Figure 3 from the paper.

    Args:
        trackers:  Dict mapping method_name → MetricsTracker.
        title:     Plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, tracker in trackers.items():
        rounds, mean_acc, ci_low, ci_high = tracker.get_convergence_history()
        if len(rounds) == 0:
            continue

        color = METHOD_COLORS.get(method_name, "#666666")

        ax.plot(rounds, mean_acc * 100, label=method_name, color=color,
                linewidth=2)
        ax.fill_between(rounds, ci_low * 100, ci_high * 100,
                        alpha=0.15, color=color)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Plot] Saved convergence plot to {save_path}")
    plt.close(fig)


def plot_communication_cost(
    method_costs: Dict[str, float],
    title: str = "Communication Cost to Reach Target Accuracy",
    save_path: Optional[str] = None,
):
    """
    Plot communication cost comparison as a horizontal lollipop chart.

    Matches Figure 4 from the paper.

    Args:
        method_costs: Dict mapping method_name → total MB transmitted.
        title:        Plot title.
        save_path:    If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = list(method_costs.keys())
    costs = list(method_costs.values())

    # Sort by cost
    sorted_pairs = sorted(zip(costs, methods))
    costs, methods = zip(*sorted_pairs)

    colors = [METHOD_COLORS.get(m, "#666666") for m in methods]
    y_pos = np.arange(len(methods))

    # Lollipop chart
    ax.hlines(y=y_pos, xmin=0, xmax=costs, color=colors, linewidth=2)
    ax.scatter(costs, y_pos, color=colors, s=100, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Total Communication Cost (MB)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate values
    for i, (cost, method) in enumerate(zip(costs, methods)):
        ax.annotate(f"{cost:.0f} MB", xy=(cost, i),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=9, va="center")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Plot] Saved comm cost plot to {save_path}")
    plt.close(fig)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    title: str = "Anomaly Detection ROC Curve",
    save_path: Optional[str] = None,
):
    """
    Plot ROC curve for anomaly detection evaluation.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auc: Area Under the Curve.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, color="#1a9988", linewidth=2,
            label=f"FLGSS (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Plot] Saved ROC plot to {save_path}")
    plt.close(fig)


def plot_ablation_bar(
    variants: List[str],
    accuracies: List[float],
    title: str = "Ablation Study",
    ylabel: str = "Test Accuracy (%)",
    save_path: Optional[str] = None,
):
    """
    Plot ablation study results as a bar chart.

    Args:
        variants:   List of variant names.
        accuracies: Corresponding accuracy values.
        title:      Plot title.
        ylabel:     Y-axis label.
        save_path:  If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#1a9988" if "FLGSS" in v or "Full" in v else "#95a5a6"
              for v in variants]

    bars = ax.bar(range(len(variants)), accuracies, color=colors,
                  edgecolor="white", linewidth=0.8)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate bars
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f"{acc:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Plot] Saved ablation plot to {save_path}")
    plt.close(fig)


def print_results_table(trackers: Dict[str, MetricsTracker]):
    """
    Print a formatted results table matching Table IV from the paper.
    """
    print(f"\n{'='*75}")
    print(f"{'Method':<16} {'Accuracy (%)':<22} {'Total Comm (MB)':<18} {'Comm to 70% (MB)':<18}")
    print(f"{'='*75}")

    for method_name, tracker in trackers.items():
        summary = tracker.get_summary()
        acc = summary["accuracy_mean"]
        ci = summary["accuracy_ci"]
        comm = summary["comm_cost_mb_mean"]
        comm_70 = summary.get("comm_to_70pct_mb", comm)

        print(f"{method_name:<16} "
              f"{acc:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]   "
              f"{comm:<18.1f} "
              f"{comm_70:<18.1f}")

    print(f"{'='*75}\n")
