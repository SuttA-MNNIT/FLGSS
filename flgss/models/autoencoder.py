import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for vision anomaly detection baseline (FedAE).
    """
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 4x4
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # 8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),   # 32x32
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon = self(x)
        # Mean squared error per image
        err = F.mse_loss(recon, x, reduction='none').flatten(1).mean(dim=1)
        return err

class SensorAutoencoder(nn.Module):
    """
    Autoencoder for tabular / sensor anomaly detection baseline (FedAE).
    """
    def __init__(self, in_features: int = 4):
        super(SensorAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 8),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, in_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon = self(x)
        err = F.mse_loss(recon, x, reduction='none').flatten(1).mean(dim=1)
        return err
