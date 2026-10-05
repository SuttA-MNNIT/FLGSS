import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Generator

from .config import FLConfig
from .models.anchor import get_anchor_model
from .models.adapter import AdapterNetwork
from .aggregation import GlobalGMM
from .defense import (
    apply_ledoit_wolf_shrinkage,
    compute_geometric_median,
    compute_mahalanobis_anomaly_score,
    inject_byzantine_attack
)
from .privacy import apply_client_level_dp
from .datasets.partition import IndexedDataset

def extract_client_statistics(z_tensor: torch.Tensor,
                              y_tensor: torch.Tensor,
                              num_classes: int,
                              latent_dim: int,
                              reg_eps: float = 1e-5) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Computes class-conditional empirical mean, covariance, and sample count:
    mu_{k,c} = (1 / |Z_{k,c}|) sum z
    Sigma_{k,c} = (1 / (|Z_{k,c}| - 1)) sum (z - mu)(z - mu)^T
    """
    stats = {}
    eye = torch.eye(latent_dim, device=z_tensor.device, dtype=z_tensor.dtype) * reg_eps

    for c in range(num_classes):
        mask = (y_tensor == c)
        count = mask.sum().item()
        if count == 0:
            continue

        z_c = z_tensor[mask]
        mean = z_c.mean(dim=0)
        
        if count > 1:
            diff = z_c - mean.unsqueeze(0)
            cov = (diff.T @ diff) / (count - 1) + eye
        else:
            cov = eye * 10.0

        stats[c] = (mean, cov, count)

    return stats

def client_flgss_update(client_id: int,
                        dataset,
                        client_indices: np.ndarray,
                        anchor_model: nn.Module,
                        cfg: FLConfig,
                        cached_anchor_embeddings: torch.Tensor = None,
                        is_malicious: bool = False) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    On-device execution for client k:
    1. Distill lightweight Adapter A_{phi_k} to match frozen Anchor E_A (Calibration phase, Eq. 1).
    2. Extract semantic space representations Z_{k,c}.
    3. Compute statistical abstraction (mean, covariance, count).
    4. Apply Client-Level DP or Byzantine Attack if configured.
    """
    if len(client_indices) == 0:
        return {}

    device = cfg.device
    client_sub = Subset(dataset, client_indices)
    indexed_sub = IndexedDataset(client_sub)

    loader = DataLoader(
        indexed_sub,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    # 1. Distillation / Adapter Calibration Phase (Eq. 1)
    adapter = AdapterNetwork(cfg.dataset, latent_dim=cfg.latent_dim, hidden_dim=cfg.adapter_hidden_dim).to(device)
    optimizer = optim.Adam(adapter.parameters(), lr=cfg.learning_rate)
    mse_loss = nn.MSELoss()

    adapter.train()
    for _ in range(cfg.local_epochs):
        for batch_data, _, batch_idx in loader:
            batch_data = batch_data.to(device)

            if cached_anchor_embeddings is not None:
                target_emb = cached_anchor_embeddings[batch_idx].to(device)
            else:
                with torch.no_grad():
                    target_emb = anchor_model(batch_data)

            optimizer.zero_grad(set_to_none=True)
            if cfg.use_amp and device == "cuda":
                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    out_emb = adapter(batch_data)
                    loss = mse_loss(out_emb, target_emb)
            else:
                out_emb = adapter(batch_data)
                loss = mse_loss(out_emb, target_emb)

            loss.backward()
            optimizer.step()

    # 2. Local Statistical Abstraction in Anchor Semantic Space (Eq. 2)
    # The statistical abstraction represents the true semantic space of the foundation model
    if cached_anchor_embeddings is not None:
        z_all = cached_anchor_embeddings.to(device).float()
        if hasattr(dataset, 'targets'):
            y_all = torch.tensor([dataset.targets[i] for i in client_indices], device=device).long()
        elif hasattr(dataset, 'labels'):
            y_all = torch.tensor([dataset.labels[i] for i in client_indices], device=device).long()
        else:
            y_all = torch.tensor([dataset[i][1] for i in client_indices], device=device).long()
    else:
        all_z, all_y = [], []
        eval_loader = DataLoader(client_sub, batch_size=cfg.test_batch_size, shuffle=False)
        with torch.no_grad():
            for batch_data, batch_y in eval_loader:
                batch_data = batch_data.to(device)
                z = anchor_model(batch_data)
                all_z.append(z)
                all_y.append(batch_y.to(device))
        z_all = torch.cat(all_z, dim=0).float()
        y_all = torch.cat(all_y, dim=0).long()

    client_stats = extract_client_statistics(z_all, y_all, cfg.num_classes, cfg.latent_dim, reg_eps=cfg.gmm_reg_covar)

    # 3. Client-Level DP (if enabled)
    if cfg.enable_dp:
        client_stats = apply_client_level_dp(
            client_stats,
            epsilon=cfg.dp_epsilon,
            delta=cfg.dp_delta,
            clip_mean=cfg.dp_clip_mean,
            clip_cov=cfg.dp_clip_cov,
            eps_psd=cfg.eps_psd
        )

    # 4. Byzantine Attack (if client is malicious)
    if is_malicious and cfg.attack_type != "none":
        client_stats = inject_byzantine_attack(
            client_stats,
            attack_type=cfg.attack_type,
            noise_scale=cfg.attack_noise_scale
        )

    return client_stats

def run_flgss(train_dataset,
              test_loader,
              client_partitions: List[np.ndarray],
              cfg: FLConfig) -> Generator[Tuple[int, Dict[str, float]], None, None]:
    """
    Main orchestration loop for FLGSS (Algorithm 1).
    Yields (round_idx, metrics_dict).
    """
    device = cfg.device
    anchor_model, latent_dim = get_anchor_model(
        dataset=cfg.dataset,
        backbone_type=cfg.backbone_type,
        pretrained=cfg.pretrained,
        latent_dim=cfg.latent_dim,
        device=device
    )

    global_gmm = GlobalGMM(num_classes=cfg.num_classes, latent_dim=latent_dim, device=device)

    # Precompute anchor cache per client for ultra-fast H100 simulation
    anchor_cache: Dict[int, torch.Tensor] = {}
    if cfg.cache_anchor:
        anchor_model.eval()
        with torch.no_grad():
            for cid in range(cfg.num_clients):
                indices = client_partitions[cid]
                if len(indices) == 0:
                    continue
                c_loader = DataLoader(Subset(train_dataset, indices), batch_size=cfg.test_batch_size, shuffle=False)
                c_embs = []
                for x, _ in c_loader:
                    x = x.to(device)
                    c_embs.append(anchor_model(x).cpu())
                anchor_cache[cid] = torch.cat(c_embs, dim=0)

    # Malicious clients if attack is enabled
    num_attackers = int(cfg.num_clients * cfg.attacker_ratio)
    malicious_clients = set(random.sample(range(cfg.num_clients), num_attackers))

    clients_per_round = max(1, int(cfg.num_clients * cfg.client_fraction))

    for r in range(cfg.num_rounds):
        round_start = time.time()
        selected_clients = random.sample(range(cfg.num_clients), clients_per_round)

        client_updates = []
        rejected_count = 0

        # Phase 1 & 2: Client Execution
        for cid in selected_clients:
            indices = client_partitions[cid]
            if len(indices) == 0:
                continue

            c_cache = anchor_cache.get(cid, None)
            is_mal = (cid in malicious_clients)

            stats = client_flgss_update(
                client_id=cid,
                dataset=train_dataset,
                client_indices=indices,
                anchor_model=anchor_model,
                cfg=cfg,
                cached_anchor_embeddings=c_cache,
                is_malicious=is_mal
            )
            if stats:
                client_updates.append((cid, stats))

        # Phase 3: Edge Server Robust Filtering & Aggregation
        accepted_updates = []
        if cfg.robust_filtering and global_gmm.is_initialized():
            for cid, stats in client_updates:
                score = compute_mahalanobis_anomaly_score(
                    client_stats=stats,
                    server_means=global_gmm.server_means,
                    server_covs=global_gmm.server_covs,
                    gamma=cfg.gmm_shrinkage
                )
                if score <= cfg.robust_threshold:
                    accepted_updates.append((cid, stats))
                else:
                    rejected_count += 1
        elif cfg.robust_filtering and r == 0:
            # Round 1 Bootstrapping via Geometric Median
            all_means_by_class = {c: [] for c in range(cfg.num_classes)}
            for cid, stats in client_updates:
                for c, (m, _, _) in stats.items():
                    all_means_by_class[c].append(m)

            class_medians = {}
            for c, m_list in all_means_by_class.items():
                if len(m_list) > 0:
                    class_medians[c] = compute_geometric_median(torch.stack(m_list))

            for cid, stats in client_updates:
                dists = [torch.norm(m - class_medians[c]).item() for c, (m, _, _) in stats.items() if c in class_medians]
                avg_dist = np.mean(dists) if dists else 0.0
                # Scale threshold for Euclidean distance on raw means
                if avg_dist < 50.0:
                    accepted_updates.append((cid, stats))
                else:
                    rejected_count += 1
        else:
            accepted_updates = client_updates

        # Update Global GMM
        global_gmm.update_from_clients(accepted_updates, gamma=cfg.gmm_shrinkage)

        # Evaluation via Bayes' Rule / MAP in Semantic Space (Eq. 3)
        # Evaluation via Bayes' Rule / MAP in Semantic Space (Eq. 3)
        correct, total = 0, 0
        if test_loader is not None:
            with torch.no_grad():
                for data, targets in test_loader:
                    data = data.to(device)
                    targets = targets.to(device)
                    z = anchor_model(data)
                    preds = global_gmm.predict(z)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
            test_acc = 100.0 * correct / max(1, total)
        else:
            test_acc = 0.0

        round_duration = time.time() - round_start

        metrics = {
            "round": r + 1,
            "test_acc": test_acc,
            "accepted_clients": len(accepted_updates),
            "rejected_clients": rejected_count,
            "duration": round_duration,
            "gmm": global_gmm
        }

        yield r + 1, metrics
