import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class SensorEncoder(nn.Module):
    """
    Lightweight Encoder for tabular and time-series sensor data (UCI-HAR, Intel, N-BaIoT).
    """
    def __init__(self, in_features: int, latent_dim: int = 128):
        super(SensorEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        return self.encoder(x)

def train_sensor_anchor(dataset_name: str, in_features: int, latent_dim: int, num_classes: int, device: str = "cuda"):
    """
    Trains and caches a domain anchor encoder for sensor datasets (UCI-HAR, Intel Lab, N-BaIoT).
    As described in Section III-A & IV-E, the sensor anchor defines the shared semantic space.
    """
    os.makedirs("./data", exist_ok=True)
    cache_path = f"./data/anchor_{dataset_name}.pt"

    encoder = SensorEncoder(in_features=in_features, latent_dim=latent_dim).to(device)

    if os.path.exists(cache_path):
        try:
            encoder.load_state_dict(torch.load(cache_path, map_location=device, weights_only=True))
            encoder.eval()
            return encoder
        except Exception:
            pass

    # Quick pre-training on sensor dataset representations
    print(f"Calibrating domain Anchor Model for {dataset_name.upper()}...")
    try:
        from ..datasets import get_uci_har_data, get_intel_lab_data, get_nbaiot_data
        if dataset_name == "uci_har":
            train_ds, _, _ = get_uci_har_data("./data")
        elif dataset_name == "intel_lab":
            train_ds, _, _ = get_intel_lab_data("./data")
        elif dataset_name == "nbaiot":
            train_ds, _, _ = get_nbaiot_data("./data")
        else:
            return encoder

        X = train_ds.features.to(device)
        y = train_ds.labels.to(device)

        head = nn.Linear(latent_dim, num_classes).to(device)
        model = nn.Sequential(encoder, head)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()

        batch_sz = min(512, len(X))
        model.train()
        for ep in range(25):
            indices = torch.randperm(len(X))
            for i in range(0, len(X), batch_sz):
                idx = indices[i:i + batch_sz]
                batch_x, batch_y = X[idx], y[idx]
                optimizer.zero_grad(set_to_none=True)
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        torch.save(encoder.state_dict(), cache_path)
        print(f"Anchor model for {dataset_name.upper()} calibrated and saved to '{cache_path}'.")
    except Exception as e:
        print(f"Notice during sensor anchor pre-training: {e}")

    encoder.eval()
    return encoder

def get_anchor_model(dataset: str, backbone_type: str = "resnet18", pretrained: bool = True, latent_dim: int = 512, device: str = "cuda"):
    """
    Builds the Anchor Model E_A.
    - ResNet-18 (ImageNet weights) for CIFAR-10.
    - Pre-trained Sensor Encoder for UCI-HAR, Intel Lab, N-BaIoT.
    """
    if dataset == "cifar10":
        if backbone_type == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            if latent_dim != in_features:
                model.fc = nn.Linear(in_features, latent_dim)
            else:
                model.fc = nn.Identity()
            out_dim = latent_dim
        elif backbone_type in ["vit", "vit_b_16", "transformer"]:
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            model = models.vit_b_16(weights=weights)
            in_features = model.heads.head.in_features
            if latent_dim != in_features:
                model.heads = nn.Linear(in_features, latent_dim)
            else:
                model.heads = nn.Identity()
            out_dim = latent_dim
        elif backbone_type in ["swin", "swin_t"]:
            weights = models.Swin_T_Weights.DEFAULT if pretrained else None
            model = models.swin_t(weights=weights)
            in_features = model.head.in_features
            if latent_dim != in_features:
                model.head = nn.Linear(in_features, latent_dim)
            else:
                model.head = nn.Identity()
            out_dim = latent_dim
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            if latent_dim != in_features:
                model.fc = nn.Linear(in_features, latent_dim)
            else:
                model.fc = nn.Identity()
            out_dim = latent_dim
    elif dataset == "uci_har":
        if pretrained:
            model = train_sensor_anchor("uci_har", in_features=561, latent_dim=latent_dim, num_classes=6, device=device)
        else:
            model = SensorEncoder(in_features=561, latent_dim=latent_dim)
        out_dim = latent_dim
    elif dataset == "intel_lab":
        if pretrained:
            model = train_sensor_anchor("intel_lab", in_features=4, latent_dim=latent_dim, num_classes=4, device=device)
        else:
            model = SensorEncoder(in_features=4, latent_dim=latent_dim)
        out_dim = latent_dim
    elif dataset == "nbaiot":
        if pretrained:
            model = train_sensor_anchor("nbaiot", in_features=115, latent_dim=latent_dim, num_classes=11, device=device)
        else:
            model = SensorEncoder(in_features=115, latent_dim=latent_dim)
        out_dim = latent_dim
    else:
        raise ValueError(f"Unknown dataset '{dataset}' for anchor model.")

    # Freeze anchor parameters
    for p in model.parameters():
        p.requires_grad = False

    model = model.to(device)
    model.eval()
    return model, out_dim
