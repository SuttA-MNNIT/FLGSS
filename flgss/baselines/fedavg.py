import copy
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Generator, Tuple, Dict, List

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

def client_train_fedavg(model: nn.Module,
                        anchor_model: nn.Module,
                        dataset,
                        indices,
                        cfg: FLConfig) -> Dict[str, torch.Tensor]:
    model.train()
    loader = DataLoader(Subset(dataset, indices), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

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

    return {k: v.cpu().detach() for k, v in model.state_dict().items()}

def run_fedavg(train_dataset,
               test_loader,
               client_partitions: List,
               cfg: FLConfig) -> Generator[Tuple[int, Dict[str, float]], None, None]:
    """
    Standard Federated Averaging (McMahan et al. 2017).
    """
    anchor_model, global_head = create_model(cfg)
    clients_per_round = max(1, int(cfg.num_clients * cfg.client_fraction))

    for r in range(cfg.num_rounds):
        round_start = time.time()
        selected_clients = random.sample(range(cfg.num_clients), clients_per_round)

        local_weights = []
        for cid in selected_clients:
            indices = client_partitions[cid]
            if len(indices) == 0:
                continue
            client_head = copy.deepcopy(global_head)
            w = client_train_fedavg(client_head, anchor_model, train_dataset, indices, cfg)
            local_weights.append((w, len(indices)))

        if not local_weights:
            continue

        # FedAvg Aggregation
        total_samples = sum(s for _, s in local_weights)
        new_state = {}
        for key in global_head.state_dict().keys():
            new_state[key] = sum(w[key] * (s / total_samples) for w, s in local_weights)

        global_head.load_state_dict({k: v.to(cfg.device) for k, v in new_state.items()})

        # Evaluate
        global_head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(cfg.device), targets.to(cfg.device)
                if anchor_model is not None:
                    z = anchor_model(data)
                    out = global_head(z)
                else:
                    out = global_head(data)
                preds = out.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        test_acc = 100.0 * correct / max(1, total)
        round_duration = time.time() - round_start

        yield r + 1, {
            "round": r + 1,
            "test_acc": test_acc,
            "duration": round_duration
        }
