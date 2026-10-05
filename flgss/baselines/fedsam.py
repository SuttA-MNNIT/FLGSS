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

def client_train_fedsam(model: nn.Module,
                        anchor_model: nn.Module,
                        dataset,
                        indices,
                        cfg: FLConfig) -> Dict[str, torch.Tensor]:
    model.train()
    loader = DataLoader(Subset(dataset, indices), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    rho = cfg.sam_rho

    for _ in range(cfg.local_epochs):
        for data, targets in loader:
            data, targets = data.to(cfg.device), targets.to(cfg.device)

            if anchor_model is not None:
                with torch.no_grad():
                    z = anchor_model(data)
                inputs = z
            else:
                inputs = data

            # First forward-backward pass to get gradient direction
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Compute perturbation eps = rho * grad / ||grad||
            grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in model.parameters() if p.grad is not None]), p=2)
            scale = rho / (grad_norm + 1e-12)

            eps_cache = []
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        eps = p.grad * scale
                        p.add_(eps)
                        eps_cache.append(eps)
                    else:
                        eps_cache.append(None)

            # Second forward-backward pass at perturbed weights
            optimizer.zero_grad(set_to_none=True)
            outputs_perturbed = model(inputs)
            loss_perturbed = criterion(outputs_perturbed, targets)
            loss_perturbed.backward()

            # Restore original weights
            with torch.no_grad():
                for p, eps in zip(model.parameters(), eps_cache):
                    if eps is not None:
                        p.sub_(eps)

            # Take optimizer step using perturbed gradient
            optimizer.step()

    return {k: v.cpu().detach() for k, v in model.state_dict().items()}

def run_fedsam(train_dataset,
               test_loader,
               client_partitions: List,
               cfg: FLConfig) -> Generator[Tuple[int, Dict[str, float]], None, None]:
    """
    FedSAM: Sharpness-Aware Minimization Federated Learning (Qu et al. 2023).
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
            w = client_train_fedsam(client_head, anchor_model, train_dataset, indices, cfg)
            local_weights.append((w, len(indices)))

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
