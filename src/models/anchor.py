"""
Frozen Anchor Model
====================
The pre-trained foundation model used as a universal encoder to project raw
data into the shared semantic latent space Z.

Key properties (Section IV-A of the paper):
  - Completely FROZEN: no parameters are updated during federation.
  - Serves as a universal transform: raw data → semantic embeddings.
  - Vision (CIFAR-10): Pre-trained ResNet-18 (ImageNet weights), FC layer removed (d=512).
  - Sensor (UCI-HAR, Intel Lab, N-BaIoT): Transformer encoder (d=128).

Fair baseline architecture:
  All baseline algorithms use the exact same frozen Anchor backbone.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from config import ModelConfig


class ResNetAnchor(nn.Module):
    """
    Frozen ResNet-based anchor for vision tasks (CIFAR-10).
    """

    def __init__(self, model_name: str = "resnet18", output_dim: int = 512):
        super().__init__()

        # Load pre-trained backbone
        if model_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feature_dim = 512
        elif model_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet variant: {model_name}")

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        if feature_dim != output_dim:
            self.projection = nn.Linear(feature_dim, output_dim)
        else:
            self.projection = nn.Identity()

        self.output_dim = output_dim
        self._freeze()

    def _freeze(self):
        """Set all parameters to non-trainable."""
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        z = self.projection(features)
        return z


class TransformerAnchor(nn.Module):
    """
    Frozen Transformer-based anchor for sensor/time-series tasks.
    """

    def __init__(
        self,
        input_dim: int = 561,
        output_dim: int = 128,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_projection = nn.Linear(d_model, output_dim)
        self.layer_norm = nn.LayerNorm(d_model)
        self._freeze()

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def _unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.input_projection(x).unsqueeze(1)
        h = self.transformer(h)
        h = self.layer_norm(h.squeeze(1))
        z = self.output_projection(h)
        return z


def pretrain_transformer_anchor(
    anchor: TransformerAnchor,
    train_dataset,
    epochs: int = 5,
    lr: float = 0.001,
    batch_size: int = 64,
    device: str = "cpu",
) -> TransformerAnchor:
    """
    Pre-train the Transformer anchor using self-supervised reconstruction.
    """
    from torch.utils.data import DataLoader

    anchor = anchor.to(device)
    anchor._unfreeze()
    anchor.train()

    decoder = nn.Linear(anchor.output_dim, anchor.input_dim).to(device)
    all_params = list(anchor.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    criterion = nn.MSELoss()

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for features, _ in loader:
            features = features.to(device)
            z = anchor(features)
            reconstructed = decoder(z)
            loss = criterion(reconstructed, features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    anchor._freeze()
    anchor.eval()
    print(f"[Anchor] Sensor Transformer anchor ({anchor.input_dim} -> {anchor.output_dim}) pre-trained & frozen.")
    return anchor


def create_anchor(config: ModelConfig, device: str = "cpu") -> nn.Module:
    """
    Factory function to create the appropriate frozen Anchor Model.
    """
    if config.anchor_type in ("resnet18", "resnet50"):
        anchor = ResNetAnchor(
            model_name=config.anchor_type,
            output_dim=config.anchor_output_dim,
        )
    elif config.anchor_type == "transformer":
        anchor = TransformerAnchor(
            input_dim=config.transformer_input_dim,
            output_dim=config.anchor_output_dim,
            d_model=config.transformer_hidden_dim,
            num_heads=config.transformer_num_heads,
            num_layers=config.transformer_num_layers,
        )
    else:
        raise ValueError(f"Unknown anchor type: {config.anchor_type}")

    return anchor.to(device)
