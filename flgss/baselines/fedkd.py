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
from .fedavg import create_model

def client_train_fedkd(global_head: nn.Module,
                       personalized_head: nn.Module,
                       anchor_model: nn.Module,
                       dataset,
                       indices,
                       cfg: FLConfig) -> Tuple[Dict[str, torch.Tensor], nn.Module]:
    global_head.train()
    personalized_head.train()

    loader = DataLoader(Subset(dataset, indices), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    opt_global = optim.Adam(global_head.parameters(), lr=cfg.learning_rate)
    opt_pers = optim.Adam(personalized_head.parameters(), lr=cfg.learning_rate)
    criterion_ce = nn.CrossEntropyLoss()
    T = cfg.kd_temperature
    alpha = cfg.kd_alpha

    for _ in range(cfg.local_epochs):
        for data, targets in loader:
            data, targets = data.to(cfg.device), targets.to(cfg.device)

            if anchor_model is not None:
                with torch.no_grad():
                    inputs = anchor_model(data)
            else:
                inputs = data

            opt_global.zero_grad(set_to_none=True)
            opt_pers.zero_grad(set_to_none=True)

            logits_g = global_head(inputs)
            logits_p = personalized_head(inputs)

            # Cross entropy losses
            loss_ce_g = criterion_ce(logits_g, targets)
            loss_ce_p = criterion_ce(logits_p, targets)

            # Mutual distillation between global and personalized representations
            loss_kd_g = F.kl_div(
                F.log_softmax(logits_g / T, dim=1),
                F.softmax(logits_p.detach() / T, dim=1),
                reduction='batchmean'
            ) * (T * T)

            loss_kd_p = F.kl_div(
                F.log_softmax(logits_p / T, dim=1),
                F.softmax(logits_g.detach() / T, dim=1),
                reduction='batchmean'
            ) * (T * T)

            loss_g = loss_ce_g + alpha * loss_kd_g
            loss_p = loss_ce_p + alpha * loss_kd_p

            loss_g.backward()
            opt_global.step()

            loss_p.backward()
            opt_pers.step()

    state_dict = {k: v.cpu().detach() for k, v in global_head.state_dict().items()}
    return state_dict, personalized_head

def run_fedkd(train_dataset,
              test_loader,
              client_partitions: List,
              cfg: FLConfig) -> Generator[Tuple[int, Dict[str, float]], None, None]:
    """
    FedKD / PFedKD: Knowledge Distillation based Federated Learning (Wu et al. 2022, Li et al. 2025).
    """
    anchor_model, global_head = create_model(cfg)
    clients_per_round = max(1, int(cfg.num_clients * cfg.client_fraction))

    # Persistent personalized client models
    personalized_models = {
        cid: copy.deepcopy(global_head).to(cfg.device) for cid in range(cfg.num_clients)
    }

    for r in range(cfg.num_rounds):
        round_start = time.time()
        selected_clients = random.sample(range(cfg.num_clients), clients_per_round)

        local_weights = []
        for cid in selected_clients:
            indices = client_partitions[cid]
            if len(indices) == 0:
                continue
            client_g = copy.deepcopy(global_head)
            client_p = personalized_models[cid]

            w, updated_p = client_train_fedkd(
                client_g,
                client_p,
                anchor_model,
                train_dataset,
                indices,
                cfg
            )
            local_weights.append((w, len(indices)))
            personalized_models[cid] = updated_p

        if not local_weights:
            continue

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
