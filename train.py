import os
import argparse
import random
import time
import numpy as np
import torch
from tqdm import tqdm

from flgss.config import FLConfig
from flgss.datasets import get_dataset, partition_data_non_iid
from flgss.flgss_core import run_flgss
from flgss.baselines import get_baseline_runner
from flgss.utils.logger import CSVLogger
from flgss.utils.profiling import calculate_communication_cost_bytes, get_peak_gpu_memory_mb, estimate_on_device_energy_joules

def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def parse_args():
    parser = argparse.ArgumentParser(description="FLGSS & Baselines Training Runner (H100 / A100 Accelerated)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "uci_har", "intel_lab", "nbaiot"])
    parser.add_argument("--algo", type=str, default="flgss",
                        choices=["flgss", "fedavg", "fedprox", "scaffold", "moon", "fedsam", "fedkd", "fedclustering", "pfedkd"])
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet Non-IID parameter (0.1=High, 1.0=Med, 10.0=Low)")
    parser.add_argument("--rounds", type=int, default=100, help="Total communication rounds")
    parser.add_argument("--clients", type=int, default=100, help="Total clients in network")
    parser.add_argument("--C", type=float, default=0.1, help="Fraction of clients per round")
    parser.add_argument("--epochs", type=int, default=5, help="Local epochs per round")
    parser.add_argument("--batch", type=int, default=32, help="Local batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal parameter")
    parser.add_argument("--latent_dim", type=int, default=None, help="Latent dimension (default 512 for vision, 128 for sensor)")
    
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50", "vit", "swin_t"],
                        help="Anchor backbone model (e.g. resnet18, resnet50, vit, swin_t)")
    parser.add_argument("--fair_backbone", action="store_true", default=True, help="Equip baselines with frozen Anchor backbone")
    parser.add_argument("--scratch_backbone", action="store_false", dest="fair_backbone", help="Train baseline CNN from scratch")
    parser.add_argument("--cache_anchor", action="store_true", default=True, help="Precompute anchor embeddings for fast H100 simulation")
    parser.add_argument("--no_cache_anchor", action="store_false", dest="cache_anchor")
    parser.add_argument("--no_anchor", action="store_true", help="Ablation: train encoder from scratch without ImageNet anchor")

    # Defense & Trustworthiness
    parser.add_argument("--robust", action="store_true", default=False, help="Enable Mahalanobis anomaly filtering")
    parser.add_argument("--no_robust", action="store_false", dest="robust", help="Disable robust filtering")
    parser.add_argument("--attack", type=str, default="none", choices=["none", "label_flip", "noise", "gaussian_shift"])
    parser.add_argument("--attacker_ratio", type=float, default=0.0, help="Fraction of Byzantine malicious clients")

    # Privacy
    parser.add_argument("--enable_dp", action="store_true", help="Enable Client-Level Differential Privacy")
    parser.add_argument("--dp_epsilon", type=float, default=2.0, help="Privacy budget epsilon")

    # Paths & System
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./runs")
    parser.add_argument("--exp", type=str, default=None, help="Custom experiment name")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # Enable H100 TF32 TensorCore acceleration
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfg = FLConfig(
        dataset=args.dataset,
        algo=args.algo,
        alpha=args.alpha,
        num_rounds=args.rounds,
        num_clients=args.clients,
        client_fraction=args.C,
        local_epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        mu=args.mu,
        backbone_type=args.backbone,
        fair_backbone=args.fair_backbone,
        pretrained=not args.no_anchor,
        cache_anchor=args.cache_anchor,
        robust_filtering=args.robust,
        attack_type=args.attack,
        attacker_ratio=args.attacker_ratio,
        enable_dp=args.enable_dp,
        dp_epsilon=args.dp_epsilon,
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed
    )

    if args.latent_dim is not None:
        cfg.latent_dim = args.latent_dim

    exp_name = args.exp or f"{cfg.dataset}_{cfg.algo}_alpha{cfg.alpha}"
    if cfg.attack_type != "none":
        exp_name += f"_attack_{cfg.attack_type}_{cfg.attacker_ratio}"
    if cfg.enable_dp:
        exp_name += f"_dp_eps{cfg.dp_epsilon}"

    log_dir = os.path.join(cfg.output_dir, exp_name)
    logger = CSVLogger(log_dir, "metrics.csv")

    print("\n" + "=" * 65)
    print(f"  FLGSS FEDERATED LEARNING ENGINE")
    print(f"  Dataset: {cfg.dataset.upper()} | Algorithm: {cfg.algo.upper()} | Alpha: {cfg.alpha}")
    print(f"  Device: {cfg.device} | H100 TF32 Acceleration: Active")
    print(f"  Log Directory: {log_dir}")
    print("=" * 65 + "\n")

    # Load Data
    print("Loading and preparing datasets...")
    train_dataset, test_loader, _ = get_dataset(cfg)
    client_partitions = partition_data_non_iid(train_dataset, cfg.num_clients, cfg.alpha, seed=cfg.seed)

    # Initialize Runner
    if cfg.algo.lower() == "flgss":
        runner = run_flgss(train_dataset, test_loader, client_partitions, cfg)
    else:
        baseline_fn = get_baseline_runner(cfg.algo)
        runner = baseline_fn(train_dataset, test_loader, client_partitions, cfg)

    # Run Simulation
    best_acc = 0.0
    pbar = tqdm(runner, total=cfg.num_rounds, desc=f"{cfg.algo.upper()} (alpha={cfg.alpha})")

    for round_num, metrics in pbar:
        test_acc = metrics.get("test_acc", 0.0)
        best_acc = max(best_acc, test_acc)

        # Profile resource metrics
        peak_gpu_mem = get_peak_gpu_memory_mb()
        energy_j = estimate_on_device_energy_joules(
            algo=cfg.algo,
            task_type="vision" if cfg.dataset == "cifar10" else "sensor",
            duration_sec=metrics.get("duration", 1.0)
        )

        full_log = {
            "round": round_num,
            "test_acc": round(test_acc, 2),
            "best_acc": round(best_acc, 2),
            "peak_gpu_mem_mb": round(peak_gpu_mem, 1),
            "energy_joules": energy_j,
            **{k: v for k, v in metrics.items() if k not in ["round", "test_acc"]}
        }

        logger.log(full_log)
        pbar.set_postfix({
            "Acc": f"{test_acc:.2f}%",
            "Best": f"{best_acc:.2f}%",
            "GPU Mem": f"{peak_gpu_mem:.0f}MB"
        })

    print(f"\nExperiment Complete! Best Accuracy: {best_acc:.2f}%")
    print(f"Metrics saved to {os.path.join(log_dir, 'metrics.csv')}\n")

if __name__ == "__main__":
    main()
