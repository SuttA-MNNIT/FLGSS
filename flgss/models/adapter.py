import torch
import torch.nn as nn

class AdapterNetwork(nn.Module):
    """
    On-device lightweight Adapter Network A_{phi_k}.
    Maps raw input x directly to the shared semantic latent space Z,
    distilled to match the Anchor Model outputs via MSE loss.
    """
    def __init__(self, dataset: str, latent_dim: int = 512, hidden_dim: int = 512):
        super(AdapterNetwork, self).__init__()
        self.dataset = dataset

        if dataset == "cifar10":
            # Compact Convolutional + Linear Adapter for image data
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # 16x16
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 8x8
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, latent_dim)
            )
        elif dataset == "uci_har":
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(561, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, latent_dim)
            )
        elif dataset == "intel_lab":
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(4, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, latent_dim)
            )
        elif dataset == "nbaiot":
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(115, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, latent_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(hidden_dim, latent_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
