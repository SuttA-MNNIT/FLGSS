"""
Lightweight Adapter Network
============================
A small MLP that learns to approximate the Anchor Model's output via
knowledge distillation (Phase 1). After training, the heavy Anchor can
be offloaded and the adapter handles all subsequent projections.

Architecture (Section IV-B, Table 2):
    Input → Dense(256) → ReLU → Dense(256) → ReLU → Dense(d)

The adapter has ~200K parameters (vs millions for the full backbone),
making it suitable for resource-constrained IoT devices.

Reference: Eq. 3 in the paper.
    φ_k* = argmin_φ E_x~D_k [ ||E_A(x) - A_φ(x)||² ]
"""

from typing import List

import torch
import torch.nn as nn

from config import ModelConfig


class AdapterNetwork(nn.Module):
    """
    Lightweight MLP adapter for knowledge distillation from the Anchor Model.

    Learns to map raw input data to the same latent space as the frozen Anchor,
    but with far fewer parameters.

    Args:
        input_dim:    Dimension of the raw input (e.g., 3×32×32=3072 for CIFAR,
                      561 for UCI-HAR).
        hidden_dims:  List of hidden layer dimensions (default: [256, 256]).
        output_dim:   Dimension of the semantic latent space d.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        output_dim: int = 512,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build MLP layers
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = h_dim

        # Final output projection (no activation — latent space is unbounded)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Count trainable parameters
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project raw input to the semantic latent space.

        Args:
            x: Input tensor. For images: (B, C, H, W); for sensors: (B, D).

        Returns:
            z: Latent vector of shape (B, output_dim).
        """
        # Flatten spatial dimensions for image inputs
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

    def get_param_count(self) -> int:
        """Return the number of trainable parameters."""
        return self.num_params


def create_adapter(
    config: ModelConfig,
    input_dim: int = None,
    device: str = "cpu",
) -> AdapterNetwork:
    """
    Factory function to create an Adapter Network.

    Args:
        config:    ModelConfig with adapter architecture settings.
        input_dim: Dimension of raw input. If None, inferred from config:
                   - CIFAR-10: 3×32×32 = 3072
                   - UCI-HAR: 561
        device:    Target device.

    Returns:
        AdapterNetwork ready for training.
    """
    if input_dim is None:
        if config.anchor_type in ("resnet18", "resnet50"):
            input_dim = 3 * 32 * 32  # CIFAR-10 flattened
        elif config.anchor_type == "transformer":
            input_dim = config.transformer_input_dim  # UCI-HAR: 561
        else:
            raise ValueError(f"Cannot infer input_dim for anchor: {config.anchor_type}")

    adapter = AdapterNetwork(
        input_dim=input_dim,
        hidden_dims=config.adapter_hidden_dims,
        output_dim=config.latent_dim,
    )

    print(f"[Adapter] Created adapter: input={input_dim}, "
          f"hidden={config.adapter_hidden_dims}, output={config.latent_dim}, "
          f"params={adapter.get_param_count():,}")

    return adapter.to(device)
