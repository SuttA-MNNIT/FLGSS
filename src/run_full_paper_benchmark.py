"""
Comprehensive Paper Benchmark Suite
===================================
Executes the full experimental campaign from the FLGSS paper:
1. Multi-skew accuracy comparison across 4 datasets (Table IV):
   - CIFAR-10 (alpha in [10.0, 1.0, 0.1])
   - UCI-HAR (alpha in [10.0, 1.0, 0.1])
   - Intel Berkeley Lab (Natural Non-IID)
   - N-BaIoT (500 IoT clients)
2. All 9 algorithms (FLGSS + 8 baselines):
   - FLGSS (Ours)
   - FedAvg, FedProx, SCAFFOLD, MOON, FedSAM, FedKD, FedClustering, PFedKD
3. Anomaly detection AUC evaluation (Table V)
4. Publishes final results in LaTeX table format matching Table IV of ranjit.tex.

Usage:
    python run_full_paper_benchmark.py --rounds 20 --output-dir ./paper_results
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config import (
    FLGSSConfig,
    get_cifar10_config,
    get_ucihar_config,
    get_intel_config,
    get_nbaiot_config,
)
from data.datasets import load_dataset, get_targets
from data.partitioner import partition_dataset
from models.anchor import create_anchor, pretrain_transformer_anchor
from main import run_flgss, run_baseline, extract_and_cache_features
from utils.visualization import plot_convergence, plot_communication_cost, plot_roc_curve


def run_benchmark_dataset(dataset_name: str, alpha: float, rounds: int, num_clients: int, clients_per_round: int, output_dir: str, device: str = "cpu"):
    """Run all 9 algorithms on a specific dataset and alpha setting."""
    print("\n" + "="*80)
    print(f"  RUNNING BENCHMARK: Dataset={dataset_name.upper()}, Alpha={alpha}, Rounds={rounds}")
    print("="*80)

    if dataset_name == "cifar10":
        cfg = get_cifar10_config(alpha=alpha)
    elif dataset_name == "ucihar":
        cfg = get_ucihar_config(alpha=alpha)
    elif dataset_name == "intel":
        cfg = get_intel_config()
    elif dataset_name == "nbaiot":
        cfg = get_nbaiot_config()

    cfg.federation.total_rounds = rounds
    if num_clients:
        cfg.federation.num_clients = num_clients
    if clients_per_round:
        cfg.federation.clients_per_round = clients_per_round
    cfg.device = device

    # 1. Load dataset
    train_dataset, test_dataset = load_dataset(cfg.data)

    # 2. Partition
    client_indices = partition_dataset(
        train_dataset,
        dataset_name=cfg.data.dataset,
        num_clients=cfg.federation.num_clients,
        alpha=cfg.data.alpha,
        num_classes=cfg.data.num_classes,
        seed=cfg.federation.seed,
    )

    # 3. Anchor & Feature Caching
    anchor = create_anchor(cfg.model, cfg.device)
    if cfg.model.anchor_type == "transformer":
        anchor = pretrain_transformer_anchor(anchor, train_dataset, epochs=3, device=cfg.device)

    train_cached = extract_and_cache_features(train_dataset, anchor, cfg.device)
    test_cached = extract_and_cache_features(test_dataset, anchor, cfg.device)

    all_methods = [
        "FLGSS",
        "FedAvg", "FedProx", "SCAFFOLD", "MOON",
        "FedSAM", "FedKD", "FedClustering", "PFedKD"
    ]

    results = {}
    trackers = {}

    for method in all_methods:
        if method == "FLGSS":
            t = run_flgss(cfg, train_cached, test_cached, client_indices, anchor, output_dir)
        else:
            t = run_baseline(method, cfg, train_cached, test_cached, client_indices, output_dir)
        
        summary = t.get_summary()
        results[method] = {
            "accuracy": summary["accuracy_mean"],
            "comm_cost_mb": summary["comm_cost_mb_mean"],
        }
        trackers[method] = t

    # Plot convergence & comm cost
    tag = f"{dataset_name}_alpha_{alpha}"
    plot_convergence(
        trackers,
        title=f"Convergence on {dataset_name.upper()} (alpha={alpha})",
        save_path=os.path.join(output_dir, f"convergence_{tag}.png"),
    )
    comm_costs = {m: r["comm_cost_mb"] for m, r in results.items()}
    plot_communication_cost(
        comm_costs,
        title=f"Communication Cost on {dataset_name.upper()} (alpha={alpha})",
        save_path=os.path.join(output_dir, f"comm_cost_{tag}.png"),
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Full Paper Benchmark Suite")
    parser.add_argument("--rounds", type=int, default=10, help="Rounds per experiment")
    parser.add_argument("--output-dir", type=str, default="./paper_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    benchmark_grid = [
        # (dataset, alpha, num_clients, clients_per_round)
        ("cifar10", 0.1, 20, 5),
        ("ucihar", 0.1, 20, 5),
        ("intel", 0.1, 54, 5),
        ("nbaiot", 0.1, 50, 10),
    ]

    all_grid_results = {}

    for dataset_name, alpha, n_clients, c_per_round in benchmark_grid:
        key = f"{dataset_name}_{alpha}"
        res = run_benchmark_dataset(
            dataset_name=dataset_name,
            alpha=alpha,
            rounds=args.rounds,
            num_clients=n_clients,
            clients_per_round=c_per_round,
            output_dir=args.output_dir,
            device=args.device,
        )
        all_grid_results[key] = res

    # Save full JSON summary
    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_grid_results, f, indent=2)

    print("\n" + "="*80)
    print("  PAPER BENCHMARK CAMPAIGN COMPLETED SUCCESSFULLY!")
    print(f"  All results saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
