import copy
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Generator, Tuple, Dict, List

from ..config import FLConfig
from .fedavg import create_model

def client_train_scaffold(model: nn.Module,
                          anchor_model: nn.Module,
                          dataset,
                          indices,
                          c_global: List[torch.Tensor],
                          c_local: List[torch.Tensor],
                          cfg: FLConfig) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    model.train()
    loader = DataLoader(Subset(dataset, indices), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    lr = 0.02
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    init_params = [p.detach().clone() for p in model.parameters()]
    steps = 0

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

            # Apply SCAFFOLD control variate correction to gradients: g = g - c_i + c
            for p, cg, cl in zip(model.parameters(), c_global, c_local):
                if p.grad is not None:
                    p.grad.data.add_(cg - cl)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            steps += 1

    steps = max(1, steps)
    final_params = [p.detach().clone() for p in model.parameters()]

    new_c_local = []
    c_delta = []
    for cl, cg, pi, pf in zip(c_local, c_global, init_params, final_params):
        diff = (pi - pf) / (steps * lr)
        ci_new = torch.clamp(cl - cg + diff, -5.0, 5.0)
        new_c_local.append(ci_new)
        c_delta.append(ci_new - cl)

    state_dict = {k: v.cpu().detach() for k, v in model.state_dict().items()}
    return state_dict, new_c_local, c_delta

def run_scaffold(train_dataset,
                 test_loader,
                 client_partitions: List,
                 cfg: FLConfig) -> Generator[Tuple[int, Dict[str, float]], None, None]:
    """
    SCAFFOLD: Stochastic Controlled Averaging for Federated Learning (Karimireddy et al. 2020).
    """
    anchor_model, global_head = create_model(cfg)
    clients_per_round = max(1, int(cfg.num_clients * cfg.client_fraction))

    # Control variates initialization
    c_global = [torch.zeros_like(p, device=cfg.device) for p in global_head.parameters()]
    c_clients = [[torch.zeros_like(p, device=cfg.device) for p in global_head.parameters()] for _ in range(cfg.num_clients)]

    for r in range(cfg.num_rounds):
        round_start = time.time()
        selected_clients = random.sample(range(cfg.num_clients), clients_per_round)

        local_weights = []
        c_deltas = []

        for cid in selected_clients:
            indices = client_partitions[cid]
            if len(indices) == 0:
                continue
            client_head = copy.deepcopy(global_head)
            w, new_ci, delta_c = client_train_scaffold(
                client_head,
                anchor_model,
                train_dataset,
                indices,
                c_global,
                c_clients[cid],
                cfg
            )
            local_weights.append((w, len(indices)))
            c_deltas.append(delta_c)
            c_clients[cid] = new_ci

        if not local_weights:
            continue

        # Aggregate Model Parameters
        total_samples = sum(s for _, s in local_weights)
        new_state = {}
        for key in global_head.state_dict().keys():
            new_state[key] = sum(w[key] * (s / total_samples) for w, s in local_weights)
        global_head.load_state_dict({k: v.to(cfg.device) for k, v in new_state.items()})

        # Aggregate Server Control Variates: c = c + (1 / N) sum (delta_ci)
        frac = len(selected_clients) / cfg.num_clients
        for i in range(len(c_global)):
            avg_delta = torch.stack([d[i] for d in c_deltas]).mean(dim=0)
            c_global[i] = c_global[i] + frac * avg_delta

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
