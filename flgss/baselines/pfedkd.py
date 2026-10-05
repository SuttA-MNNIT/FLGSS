import copy
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def client_train_pfedkd(local_model: nn.Module,
                        global_model: nn.Module,
                        anchor_model: nn.Module,
                        dataset,
                        indices,
                        cfg: FLConfig,
                        alpha_kd: float = 0.5,
                        temperature: float = 2.0) -> Dict[str, torch.Tensor]:
    """
    On-device personalized distillation for PFedKD (Li et al. 2025).
    Jointly updates local personalized head and produces global distillation update.
    """
    local_model.train()
    global_model.eval()

    loader = DataLoader(Subset(dataset, indices), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    optimizer = optim.Adam(local_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    for _ in range(cfg.local_epochs):
        for data, targets in loader:
            data, targets = data.to(cfg.device), targets.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)

            if anchor_model is not None:
                with torch.no_grad():
                    z = anchor_model(data)
            else:
                z = data

            student_logits = local_model(z)
            loss_task = ce_loss(student_logits, targets)

            with torch.no_grad():
                teacher_logits = global_model(z)

            # KL Distillation loss
            soft_targets = F.softmax(teacher_logits / temperature, dim=1)
            soft_prob = F.log_softmax(student_logits / temperature, dim=1)
            loss_kd = kl_loss(soft_prob, soft_targets) * (temperature ** 2)

            total_loss = (1.0 - alpha_kd) * loss_task + alpha_kd * loss_kd
            total_loss.backward()
            optimizer.step()

    return {k: v.cpu().detach() for k, v in local_model.state_dict().items()}

def run_pfedkd(train_dataset,
               test_loader,
               client_partitions: List,
               cfg: FLConfig) -> Generator[Tuple[int, Dict[str, float]], None, None]:
    """
    PFedKD: Personalized Federated Knowledge Distillation (Li et al. 2025).
    """
    anchor_model, global_head = create_model(cfg)
    
    # Maintain personalized models for clients
    client_heads = {cid: copy.deepcopy(global_head) for cid in range(cfg.num_clients)}
    clients_per_round = max(1, int(cfg.num_clients * cfg.client_fraction))

    for r in range(cfg.num_rounds):
        round_start = time.time()
        selected_clients = random.sample(range(cfg.num_clients), clients_per_round)

        local_weights = []
        for cid in selected_clients:
            indices = client_partitions[cid]
            if len(indices) == 0:
                continue

            w = client_train_pfedkd(
                local_model=client_heads[cid],
                global_model=global_head,
                anchor_model=anchor_model,
                dataset=train_dataset,
                indices=indices,
                cfg=cfg,
                alpha_kd=cfg.kd_alpha,
                temperature=cfg.kd_temperature
            )
            local_weights.append((w, len(indices)))

        if not local_weights:
            continue

        # Global Teacher Aggregation (FedAvg on personalized updates)
        total_samples = sum(s for _, s in local_weights)
        new_global = {}
        for key in global_head.state_dict().keys():
            new_global[key] = sum(w[key] * (s / total_samples) for w, s in local_weights)

        global_head.load_state_dict({k: v.to(cfg.device) for k, v in new_global.items()})

        # Synchronize personalization: blend global teacher back to local models
        beta = 0.5
        for cid in selected_clients:
            curr_state = client_heads[cid].state_dict()
            blended = {k: (1.0 - beta) * curr_state[k] + beta * new_global[k].to(cfg.device) for k in curr_state}
            client_heads[cid].load_state_dict(blended)

        # Evaluate global model on test set
        global_head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(cfg.device), targets.to(cfg.device)
                if anchor_model is not None:
                    z = anchor_model(data)
                else:
                    z = data
                preds = torch.argmax(global_head(z), dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        test_acc = 100.0 * correct / max(1, total)
        round_duration = time.time() - round_start

        yield r + 1, {
            "round": r + 1,
            "test_acc": test_acc,
            "duration": round_duration
        }
