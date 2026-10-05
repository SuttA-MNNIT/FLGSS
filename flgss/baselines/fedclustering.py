import copy
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Generator, Tuple, Dict, List
import numpy as np

from ..config import FLConfig
from ..models import get_anchor_model, BaselineClassificationHead, StandaloneCNN, StandaloneMLP

def create_model(cfg: FLConfig):
    if cfg.fair_backbone:
        anchor_model, latent_dim = get_anchor_model(
            dataset=cfg.dataset,
            backbone_type=cfg.backbone_type,
            pretrained=cfg.pretrained,
            latent_dim=cfg.latent_dim,
            device=cfg.device
        )
        head = BaselineClassificationHead(latent_dim=latent_dim, num_classes=cfg.num_classes).to(cfg.device)
        return anchor_model, head
    else:
        if cfg.dataset == "cifar10":
            model = StandaloneCNN(num_classes=cfg.num_classes).to(cfg.device)
        else:
            in_dim = 561 if cfg.dataset == "uci_har" else (4 if cfg.dataset == "intel_lab" else 115)
            model = StandaloneMLP(in_features=in_dim, num_classes=cfg.num_classes).to(cfg.device)
        return None, model

def client_train(model: nn.Module,
                 anchor_model: nn.Module,
                 dataset,
                 indices,
                 cfg: FLConfig) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    model.train()
    loader = DataLoader(Subset(dataset, indices), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    init_weights = torch.cat([p.flatten() for p in model.parameters()])

    for _ in range(cfg.local_epochs):
        for data, targets in loader:
            data, targets = data.to(cfg.device), targets.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)

            if anchor_model is not None:
                with torch.no_grad():
                    z = anchor_model(data)
                outputs = model(z)
            else:
                outputs = model(data)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    final_weights = torch.cat([p.flatten() for p in model.parameters()])
    delta_w = (final_weights - init_weights).detach().cpu()
    state_dict = {k: v.cpu().detach() for k, v in model.state_dict().items()}
    return state_dict, delta_w

def run_fedclustering(train_dataset,
                      test_loader,
                      client_partitions: List,
                      cfg: FLConfig) -> Generator[Tuple[int, Dict[str, float]], None, None]:
    """
    FedClustering: Clustered Federated Learning (Zhao et al. 2025).
    Partitions clients into semantic clusters based on gradient/weight updates similarity.
    """
    anchor_model, global_head = create_model(cfg)
    num_clusters = min(3, cfg.num_classes)
    
    # Initialize cluster models
    cluster_heads = [copy.deepcopy(global_head) for _ in range(num_clusters)]
    client_clusters = {cid: cid % num_clusters for cid in range(cfg.num_clients)}
    
    clients_per_round = max(1, int(cfg.num_clients * cfg.client_fraction))

    for r in range(cfg.num_rounds):
        round_start = time.time()
        selected_clients = random.sample(range(cfg.num_clients), clients_per_round)

        updates_by_cluster = {k: [] for k in range(num_clusters)}
        client_deltas = {}

        for cid in selected_clients:
            indices = client_partitions[cid]
            if len(indices) == 0:
                continue

            k = client_clusters[cid]
            client_head = copy.deepcopy(cluster_heads[k])
            w, delta = client_train(client_head, anchor_model, train_dataset, indices, cfg)
            
            updates_by_cluster[k].append((w, len(indices)))
            client_deltas[cid] = delta

        # Re-cluster clients periodically based on cosine similarity of updates
        if r > 0 and r % 5 == 0 and len(client_deltas) >= num_clusters:
            cids = list(client_deltas.keys())
            deltas_mat = torch.stack([client_deltas[c] for c in cids])
            deltas_norm = torch.nn.functional.normalize(deltas_mat, p=2, dim=1)
            sim_matrix = torch.mm(deltas_norm, deltas_norm.T).numpy()

            # Simple K-means style cluster assignment on similarity
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=cfg.seed)
            labels = kmeans.fit_predict(sim_matrix)
            for idx, cid in enumerate(cids):
                client_clusters[cid] = int(labels[idx])

        # Aggregate within each cluster
        for k in range(num_clusters):
            cluster_updates = updates_by_cluster[k]
            if not cluster_updates:
                continue
            total_samples = sum(s for _, s in cluster_updates)
            new_state = {}
            for key in cluster_heads[k].state_dict().keys():
                new_state[key] = sum(w[key] * (s / total_samples) for w, s in cluster_updates)
            cluster_heads[k].load_state_dict({key: v.to(cfg.device) for key, v in new_state.items()})

        # Evaluate ensemble of cluster heads on global test set
        for h in cluster_heads:
            h.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(cfg.device), targets.to(cfg.device)
                if anchor_model is not None:
                    z = anchor_model(data)
                else:
                    z = data
                
                # Ensemble average across clusters
                logits = torch.stack([h(z) for h in cluster_heads]).mean(dim=0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        test_acc = 100.0 * correct / max(1, total)
        round_duration = time.time() - round_start

        yield r + 1, {
            "round": r + 1,
            "test_acc": test_acc,
            "duration": round_duration
        }
