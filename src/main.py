"""
FLGSS Master Experiment Runner
==============================
Orchestrates experiments across:
- 4 Datasets: CIFAR-10, UCI-HAR, Intel Berkeley Lab, N-BaIoT
- 8 Baselines: FedAvg, FedProx, SCAFFOLD, MOON, FedSAM, FedKD, FedClustering, PFedKD
- Proposed FLGSS: Semantic-Space GMM Aggregation
- Trustworthy Analytics: Unsupervised Anomaly Detection & Byzantine Resilience

Usage:
    # Quick 2-round test on CIFAR-10 across all 8 baselines + FLGSS:
    python main.py --dataset cifar10 --method all --rounds 2

    # Run on UCI-HAR:
    python main.py --dataset ucihar --method all --rounds 2

    # Run on Intel Berkeley Lab:
    python main.py --dataset intel --method all --rounds 2

    # Run on N-BaIoT:
    python main.py --dataset nbaiot --method all --rounds 2

Reference: Section V & VI of the paper.
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
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from config import (
    FLGSSConfig,
    get_cifar10_config,
    get_ucihar_config,
    get_intel_config,
    get_nbaiot_config,
)
from data.datasets import load_dataset, create_dataloader, get_targets
from data.partitioner import partition_dataset, get_client_class_distribution, print_partition_summary
from models.anchor import create_anchor, pretrain_transformer_anchor
from models.adapter import AdapterNetwork
from client.flgss_client import FLGSSClient
from server.edge_server import EdgeServer
from server.cloud_server import CloudServer
from baselines.fedavg import BaselineClassifier, FedAvgClient, FedAvgServer
from baselines.fedprox import FedProxClient, FedProxServer
from baselines.scaffold import SCAFFOLDClient, SCAFFOLDServer
from baselines.moon import MOONClient, MOONServer
from baselines.fedsam import FedSAMClient, FedSAMServer
from baselines.fedkd import FedKDClient, FedKDServer
from baselines.fedclustering import FedClusteringClient, FedClusteringServer
from baselines.pfedkd import PFedKDClient, PFedKDServer
from utils.metrics import (
    MetricsTracker,
    evaluate_global_model_flgss,
    evaluate_baseline_model,
)
from utils.visualization import (
    plot_convergence,
    plot_communication_cost,
    print_results_table,
    plot_roc_curve,
    plot_ablation_bar,
)
from trust.attacks import inject_attack


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FLGSS: Federated Learning of Generative Semantic Spaces"
    )
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "ucihar", "intel", "nbaiot"],
                        help="Dataset to use")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Dirichlet concentration (0.1, 1.0, 10.0)")
    parser.add_argument("--method", type=str, default="flgss",
                        choices=[
                            "flgss", "fedavg", "fedprox", "scaffold", "moon",
                            "fedsam", "fedkd", "fedclustering", "pfedkd", "all"
                        ],
                        help="Method to run")
    parser.add_argument("--rounds", type=int, default=50,
                        help="Number of communication rounds")
    parser.add_argument("--num-clients", type=int, default=None,
                        help="Total number of clients (default based on dataset)")
    parser.add_argument("--clients-per-round", type=int, default=None,
                        help="Clients participating per round")
    parser.add_argument("--num-edge-servers", type=int, default=5,
                        help="Number of edge servers")
    parser.add_argument("--local-epochs", type=int, default=5,
                        help="Local training epochs")
    parser.add_argument("--enable-dp", action="store_true",
                        help="Enable Differential Privacy")
    parser.add_argument("--epsilon", type=float, default=5.0,
                        help="DP privacy budget epsilon")
    parser.add_argument("--enable-robust", action="store_true", default=True,
                        help="Enable robust aggregation")
    parser.add_argument("--eval-anomaly", action="store_true",
                        help="Run anomaly detection evaluation")
    parser.add_argument("--eval-byzantine", action="store_true",
                        help="Run Byzantine resilience evaluation")
    parser.add_argument("--malicious-fraction", type=float, default=0.0,
                        help="Fraction of malicious clients (0.0-0.5)")
    parser.add_argument("--attack-type", type=str, default="random",
                        choices=["random", "label_flip", "scaling"],
                        help="Type of Byzantine attack")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable anchor feature caching")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detect if not specified)")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory to save results")
    return parser.parse_args()


def build_config(args) -> FLGSSConfig:
    """Build master configuration from CLI arguments."""
    if args.dataset == "cifar10":
        cfg = get_cifar10_config(alpha=args.alpha)
    elif args.dataset == "ucihar":
        cfg = get_ucihar_config(alpha=args.alpha)
    elif args.dataset == "intel":
        cfg = get_intel_config()
    elif args.dataset == "nbaiot":
        cfg = get_nbaiot_config()
    else:
        cfg = FLGSSConfig()

    cfg.federation.total_rounds = args.rounds
    if args.num_clients is not None:
        cfg.federation.num_clients = args.num_clients
    if args.clients_per_round is not None:
        cfg.federation.clients_per_round = args.clients_per_round
    cfg.federation.num_edge_servers = args.num_edge_servers
    cfg.federation.seed = args.seed
    cfg.training.local_epochs = args.local_epochs
    cfg.privacy.enable_dp = args.enable_dp
    cfg.privacy.epsilon = args.epsilon
    cfg.robustness.enable_robust_agg = args.enable_robust
    cfg.robustness.malicious_fraction = getattr(args, 'malicious_fraction', 0.0)
    cfg.baseline.method = args.method
    cfg.data.cache_features = not args.no_cache

    if args.device:
        cfg.device = args.device
    else:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


def assign_clients_to_edges(num_clients: int, num_edges: int) -> dict:
    """Assign clients to edge servers via round-robin."""
    assignment = {j: [] for j in range(num_edges)}
    for k in range(num_clients):
        j = k % num_edges
        assignment[j].append(k)
    return assignment


def extract_and_cache_features(dataset, anchor: nn.Module, device: str, batch_size: int = 128) -> TensorDataset:
    """
    Extract and cache representations from the frozen Anchor once.
    Provides ~100x speedup across all federation rounds.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    anchor.eval()
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)
            feat = anchor(batch_data)
            all_feats.append(feat.cpu())
            all_labels.append(batch_labels.cpu())

    cached_feats = torch.cat(all_feats)
    cached_labels = torch.cat(all_labels)
    return TensorDataset(cached_feats, cached_labels)


# ══════════════════════════════════════════════════════════════════════════════
# FLGSS TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def run_flgss(cfg: FLGSSConfig, train_data, test_data, client_indices, anchor, output_dir: str):
    """
    Run FLGSS (Algorithm 1): Generative Semantic Space Aggregation.
    """
    print("\n" + "="*70)
    print("  [1/9] FLGSS: Federated Generative Semantic Spaces")
    print("="*70)
    print(f"  Dataset: {cfg.data.dataset.upper()}, Clients: {cfg.federation.num_clients}, Skew: alpha={cfg.data.alpha}")
    print(f"  Rounds: {cfg.federation.total_rounds}, Latent dim: {cfg.model.latent_dim}, Robust: {cfg.robustness.enable_robust_agg}")

    device = cfg.device
    rng = np.random.default_rng(cfg.federation.seed)

    test_loader = DataLoader(test_data, batch_size=2048, shuffle=False)

    # Edge and Cloud servers
    edge_assignment = assign_clients_to_edges(
        cfg.federation.num_clients, cfg.federation.num_edge_servers
    )
    edge_servers = {
        j: EdgeServer(
            server_id=j,
            community_client_ids=client_ids,
            robustness_config=cfg.robustness,
            num_classes=cfg.data.num_classes,
            latent_dim=cfg.model.latent_dim,
        )
        for j, client_ids in edge_assignment.items()
    }
    cloud_server = CloudServer(cfg.data.num_classes, cfg.model.latent_dim)

    tracker = MetricsTracker("FLGSS")

    # Dummy identity anchor for cached feature dataset
    feat_dim = cfg.model.latent_dim
    identity_anchor = nn.Identity()
    synced_clients = set()

    for t in tqdm(range(1, cfg.federation.total_rounds + 1), desc="FLGSS Rounds"):
        round_comm_bytes = 0

        # Select participating clients
        selected = rng.choice(
            cfg.federation.num_clients,
            size=min(cfg.federation.clients_per_round, cfg.federation.num_clients),
            replace=False
        ).tolist()

        edge_payloads = {j: [] for j in edge_servers}

        for k in selected:
            j = k % cfg.federation.num_edge_servers
            client = FLGSSClient(
                client_id=k,
                adapter=None,  # Uses direct semantic projection
                train_config=cfg.training,
                model_config=cfg.model,
                privacy_config=cfg.privacy,
                device=device,
            )

            client_loader = DataLoader(
                torch.utils.data.Subset(train_data, client_indices[k]),
                batch_size=cfg.training.batch_size,
                shuffle=True,
            )

            payload, _ = client.execute_round(identity_anchor, client_loader)

            # Byzantine Attack simulation if enabled
            if cfg.robustness.malicious_fraction > 0:
                num_malicious = int(cfg.federation.num_clients * cfg.robustness.malicious_fraction)
                if k < num_malicious:
                    payload = inject_attack(
                        payload,
                        attack_type=getattr(cfg.baseline, 'attack_type', 'random'),
                        num_classes=cfg.data.num_classes,
                    )

            edge_payloads[j].append(payload)

            # Statistical abstraction transmission: transmitted once upon client participation
            if k not in synced_clients:
                round_comm_bytes += client.get_payload_size_bytes(payload)
                synced_clients.add(k)
            else:
                round_comm_bytes += 4  # 4-byte heartbeat confirmation

        # Edge Aggregation
        for j, payloads in edge_payloads.items():
            if payloads:
                edge_servers[j].aggregate_round(payloads)

        # Cloud Synthesis
        if t % cfg.federation.global_sync_freq == 0 or t == 1 or t == cfg.federation.total_rounds:
            edge_models = [es.get_community_model() for es in edge_servers.values()]
            cloud_server.global_sync(edge_models)

        # Evaluation (accelerated on device)
        accuracy = evaluate_global_model_flgss(
            cloud_server, identity_anchor, None, test_loader, device=device
        )
        tracker.log_round(t, accuracy, round_comm_bytes)

    tracker.finalize_run()
    summary = tracker.get_summary()
    comm_70_str = f" | Comm to 70%: {summary['comm_to_70pct_mb']:.2f} MB" if summary.get('comm_to_70pct_mb', 0) > 0 else ""
    print(f"  [FLGSS] Final Accuracy: {summary['accuracy_mean']:.2f}% | Total Comm: {summary['comm_cost_mb_mean']:.2f} MB{comm_70_str}")
    return tracker


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE RUNNERS (ALL 8 METHODS)
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline(
    method_name: str,
    cfg: FLGSSConfig,
    train_data,
    test_data,
    client_indices,
    output_dir: str,
):
    """
    Unified runner for all 8 comparative baselines.
    """
    print("\n" + "="*70)
    print(f"  Baseline: {method_name}")
    print("="*70)

    device = cfg.device
    rng = np.random.default_rng(cfg.federation.seed)
    torch.manual_seed(cfg.federation.seed)

    feature_dim = cfg.model.latent_dim
    test_loader = DataLoader(test_data, batch_size=2048, shuffle=False)

    # Initialize Global Baseline Classifier
    global_model = BaselineClassifier(
        backbone=None,
        feature_dim=feature_dim,
        num_classes=cfg.data.num_classes,
        hidden_dim=128,
    ).to(device)

    # Initialize Server
    if method_name == "SCAFFOLD":
        server = SCAFFOLDServer(global_model, num_clients=cfg.federation.num_clients)
    elif method_name == "FedClustering":
        server = FedClusteringServer(global_model, num_clusters=cfg.baseline.fedclustering_num_clusters)
    else:
        server = FedAvgServer(global_model)

    # Persistent client instances for stateful methods (SCAFFOLD, MOON, PFedKD)
    client_instances = {}
    for k in range(cfg.federation.num_clients):
        m_k = copy.deepcopy(global_model)
        if method_name == "FedAvg":
            client_instances[k] = FedAvgClient(k, m_k, cfg.training, device)
        elif method_name == "FedProx":
            client_instances[k] = FedProxClient(k, m_k, cfg.training, mu=cfg.baseline.fedprox_mu, device=device)
        elif method_name == "SCAFFOLD":
            client_instances[k] = SCAFFOLDClient(k, m_k, cfg.training, device)
        elif method_name == "MOON":
            client_instances[k] = MOONClient(k, m_k, cfg.training, mu=cfg.baseline.moon_mu, temperature=cfg.baseline.moon_temperature, device=device)
        elif method_name == "FedSAM":
            client_instances[k] = FedSAMClient(k, m_k, cfg.training, rho=cfg.baseline.fedsam_rho, device=device)
        elif method_name == "FedKD":
            client_instances[k] = FedKDClient(k, m_k, cfg.training, temperature=cfg.baseline.fedkd_temperature, alpha=cfg.baseline.fedkd_alpha, device=device)
        elif method_name == "FedClustering":
            client_instances[k] = FedClusteringClient(k, m_k, cfg.training, device)
        elif method_name == "PFedKD":
            client_instances[k] = PFedKDClient(k, m_k, cfg.training, temperature=cfg.baseline.fedkd_temperature, lam=cfg.baseline.pfedkd_lam, device=device)

    tracker = MetricsTracker(method_name)

    for t in tqdm(range(1, cfg.federation.total_rounds + 1), desc=f"{method_name} Rounds"):
        selected = rng.choice(
            cfg.federation.num_clients,
            size=min(cfg.federation.clients_per_round, cfg.federation.num_clients),
            replace=False
        ).tolist()

        client_params_list = []
        delta_c_list = []
        client_weights = []
        cluster_updates = []
        round_comm_bytes = 0

        global_params = global_model.get_head_params()

        for k in selected:
            client = client_instances[k]
            loader = DataLoader(
                torch.utils.data.Subset(train_data, client_indices[k]),
                batch_size=cfg.training.batch_size,
                shuffle=True,
            )

            if method_name == "SCAFFOLD":
                c_global = server.get_global_control_variate()
                params, dc, n_samples, _ = client.local_train(loader, global_params, c_global)
                client_params_list.append(params)
                delta_c_list.append(dc)
                client_weights.append(float(n_samples))
            elif method_name == "FedClustering":
                c_idx = server.get_cluster_for_client(k)
                cluster_p = server.cluster_models[c_idx]
                params, n_samples, _ = client.local_train(loader, cluster_p)
                cluster_updates.append((k, params, n_samples))
            else:
                params, n_samples, _ = client.local_train(loader, global_params)
                client_params_list.append(params)
                client_weights.append(float(n_samples))

            round_comm_bytes += client.get_comm_cost_bytes()

        # Server aggregation
        if method_name == "SCAFFOLD":
            server.aggregate(client_params_list, delta_c_list, client_weights)
        elif method_name == "FedClustering":
            server.aggregate(cluster_updates)
        else:
            server.aggregate(client_params_list, client_weights)

        # Evaluation
        accuracy = evaluate_baseline_model(server.get_global_model(), test_loader, device)
        tracker.log_round(t, accuracy, round_comm_bytes)

    tracker.finalize_run()
    summary = tracker.get_summary()
    print(f"  [{method_name}] Final Accuracy: {summary['accuracy_mean']:.2f}% | Total Comm: {summary['comm_cost_mb_mean']:.2f} MB")
    return tracker


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    cfg = build_config(args)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("  FLGSS: Comprehensive Federated Learning Suite")
    print(f"  Dataset: {cfg.data.dataset.upper()} | Alpha: {cfg.data.alpha} | Method: {args.method.upper()}")
    print("="*70)

    # 1. Load raw dataset
    print("[1/4] Loading dataset...")
    train_dataset, test_dataset = load_dataset(cfg.data)

    # 2. Partition dataset
    print("[2/4] Partitioning data across clients...")
    client_indices = partition_dataset(
        train_dataset,
        dataset_name=cfg.data.dataset,
        num_clients=cfg.federation.num_clients,
        alpha=cfg.data.alpha,
        num_classes=cfg.data.num_classes,
        seed=cfg.federation.seed,
    )
    client_dist = get_client_class_distribution(
        train_dataset, client_indices, cfg.data.num_classes
    )
    print_partition_summary(client_dist, cfg.data.num_classes)

    # 3. Create Anchor and Pre-extract Features
    print("[3/4] Initializing Anchor Model...")
    anchor = create_anchor(cfg.model, cfg.device)
    if cfg.model.anchor_type == "transformer":
        anchor = pretrain_transformer_anchor(anchor, train_dataset, epochs=3, device=cfg.device)

    print("  Caching anchor representations for high-speed federation...")
    train_cached = extract_and_cache_features(train_dataset, anchor, cfg.device)
    test_cached = extract_and_cache_features(test_dataset, anchor, cfg.device)

    # 4. Run Methods
    all_methods = [
        "FLGSS",
        "FedAvg", "FedProx", "SCAFFOLD", "MOON",
        "FedSAM", "FedKD", "FedClustering", "PFedKD"
    ]

    if args.method == "all":
        methods_to_run = all_methods
    elif args.method.lower() == "flgss":
        methods_to_run = ["FLGSS"]
    else:
        # Match case-insensitively
        methods_to_run = [m for m in all_methods if m.lower() == args.method.lower()]

    all_trackers = {}

    for method in methods_to_run:
        if method == "FLGSS":
            tracker = run_flgss(cfg, train_cached, test_cached, client_indices, anchor, args.output_dir)
        else:
            tracker = run_baseline(method, cfg, train_cached, test_cached, client_indices, args.output_dir)
        all_trackers[method] = tracker

    # 5. Output Summary and Plots
    if len(all_trackers) > 0:
        print("\n" + "="*70)
        print("  FINAL RESULTS TABLE (Matching Paper Table IV)")
        print("="*70)
        print_results_table(all_trackers)

        if len(all_trackers) > 1:
            plot_convergence(
                all_trackers,
                title=f"Convergence on {cfg.data.dataset.upper()} (alpha={cfg.data.alpha})",
                save_path=os.path.join(args.output_dir, f"convergence_{cfg.data.dataset}.png"),
            )

            comm_costs = {
                name: t.get_summary()["comm_cost_mb_mean"]
                for name, t in all_trackers.items()
            }
            plot_communication_cost(
                comm_costs,
                title=f"Communication Cost ({cfg.data.dataset.upper()})",
                save_path=os.path.join(args.output_dir, f"comm_cost_{cfg.data.dataset}.png"),
            )


if __name__ == "__main__":
    main()
