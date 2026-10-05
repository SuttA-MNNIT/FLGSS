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

def client_train_moon(model: nn.Module,
                      global_model: nn.Module,
                      prev_model: nn.Module,
                      anchor_model: nn.Module,
                      dataset,
                      indices,
                      cfg: FLConfig) -> Dict[str, torch.Tensor]:
    model.train()
    global_model.eval()
    if prev_model is not None:
        prev_model.eval()

    loader = DataLoader(Subset(dataset, indices), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion_ce = nn.CrossEntropyLoss()
    cos_sim = nn.CosineSimilarity(dim=-1)

    for _ in range(cfg.local_epochs):
        for data, targets in loader:
            data, targets = data.to(cfg.device), targets.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)

            if anchor_model is not None:
                with torch.no_grad():
                    z_in = anchor_model(data)
                outputs = model(z_in)
                with torch.no_grad():
                    z_glob = global_model(z_in)
                    z_prev = prev_model(z_in) if prev_model is not None else z_glob
            else:
                outputs = model(data)
                with torch.no_grad():
                    z_glob = global_model(data)
                    z_prev = prev_model(data) if prev_model is not None else z_glob

            loss_ce = criterion_ce(outputs, targets)

            # Model-Contrastive Loss (MOON)
            sim_pos = cos_sim(outputs, z_glob) / cfg.moon_temp
            sim_neg = cos_sim(outputs, z_prev) / cfg.moon_temp
            loss_con = -torch.log(torch.exp(sim_pos) / (torch.exp(sim_pos) + torch.exp(sim_neg) + 1e-8)).mean()

            total_loss = loss_ce + cfg.moon_mu * loss_con
            total_loss.backward()
            optimizer.step()

    return {k: v.cpu().detach() for k, v in model.state_dict().items()}

def run_moon(train_dataset,
             test_loader,
             client_partitions: List,
             cfg: FLConfig) -> Generator[Tuple[int, Dict[str, float]], None, None]:
    """
    MOON: Model-Contrastive Federated Learning (Li et al. 2021).
    """
    anchor_model, global_head = create_model(cfg)
    clients_per_round = max(1, int(cfg.num_clients * cfg.client_fraction))
    prev_models: Dict[int, nn.Module] = {}

    for r in range(cfg.num_rounds):
        round_start = time.time()
        selected_clients = random.sample(range(cfg.num_clients), clients_per_round)

        local_weights = []
        for cid in selected_clients:
            indices = client_partitions[cid]
            if len(indices) == 0:
                continue
            client_head = copy.deepcopy(global_head)
            prev_m = prev_models.get(cid, None)

            w = client_train_moon(
                client_head,
                global_head,
                prev_m,
                anchor_model,
                train_dataset,
                indices,
                cfg
            )
            local_weights.append((w, len(indices)))
            prev_models[cid] = copy.deepcopy(client_head)

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
