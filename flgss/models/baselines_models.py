import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineClassificationHead(nn.Module):
    """
    Downstream linear classification head for fair comparison across FL baselines (Fair Backbone Protocol).
    Operates on top of the representations extracted by the frozen Anchor Backbone:
    'Frozen Backbone -> Trainable Linear Classifier'.
    """
    def __init__(self, latent_dim: int, num_classes: int):
        super(BaselineClassificationHead, self).__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)

class StandaloneCNN(nn.Module):
    """
    Standard standalone 4-layer CNN for CIFAR-10 (used when fair_backbone=False).
    """
    def __init__(self, num_classes: int = 10):
        super(StandaloneCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class StandaloneMLP(nn.Module):
    """
    Standard standalone MLP for sensor tasks (UCI-HAR, Intel, N-BaIoT).
    """
    def __init__(self, in_features: int = 561, num_classes: int = 6):
        super(StandaloneMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        return self.net(x)
