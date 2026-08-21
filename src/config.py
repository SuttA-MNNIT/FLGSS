"""
Centralized Configuration for FLGSS Experiments
================================================
Dataclass-based config covering all hyperparameters from Table 2 of the paper:
- Dataset & partitioning: CIFAR-10 / UCI-HAR / Intel Berkeley Lab / N-BaIoT
- Architecture: anchor model, adapter dims, latent space dim d
- Training: local epochs, learning rate, batch size, total rounds
- Privacy: ε, δ, clipping norms
- Robustness: Mahalanobis threshold τ, Ledoit-Wolf shrinkage γ
- Federation: number of clients, participation fraction, edge servers
- Baselines: parameters for all 8 comparative federated methods
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Dataset and non-IID partitioning configuration."""
    dataset: str = "cifar10"                  # "cifar10", "ucihar", "intel", "nbaiot"
    data_root: str = "./data_cache"           # Where to download/store data
    alpha: float = 0.1                        # Dirichlet concentration (0.1=high skew)
    num_classes: int = 10                     # Number of target classes
    cache_features: bool = True               # Cache anchor features for fast execution


@dataclass
class ModelConfig:
    """Anchor model and adapter architecture configuration."""
    # Anchor Model
    anchor_type: str = "resnet18"             # "resnet18", "resnet50", "transformer", or "sensor_mlp"
    anchor_output_dim: int = 512              # Dimension of anchor encoder output
    freeze_anchor: bool = True                # Always True per paper design

    # Adapter Network (MLP)
    adapter_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    latent_dim: int = 512                     # d: dimension of semantic space (512 CIFAR, 128 sensor)

    # Sensor / Transformer Anchor
    transformer_input_dim: int = 561          # 561 for UCI-HAR, 8 for Intel, 115 for N-BaIoT
    transformer_num_heads: int = 4
    transformer_num_layers: int = 2
    transformer_hidden_dim: int = 256


@dataclass
class TrainingConfig:
    """Local training hyperparameters (Table 2 in paper)."""
    local_epochs: int = 5                     # E: local training epochs
    learning_rate: float = 0.001              # η: Adam learning rate
    batch_size: int = 32                      # Local batch size
    optimizer: str = "adam"                   # Optimizer type
    weight_decay: float = 0.0                 # L2 regularization


@dataclass
class FederationConfig:
    """Federated learning topology and scheduling."""
    num_clients: int = 100                    # N: total number of clients
    clients_per_round: int = 10               # Clients selected per round
    total_rounds: int = 50                    # T: total communication rounds (default 50 for experiments)
    num_edge_servers: int = 5                 # M: number of edge servers
    global_sync_freq: int = 5                 # How often edge→cloud sync occurs
    seed: int = 42                            # Random seed for reproducibility
    num_runs: int = 5                         # Independent runs for CI


@dataclass
class PrivacyConfig:
    """Client-Level Differential Privacy parameters (Section IV-F)."""
    enable_dp: bool = False                   # Toggle DP on/off
    epsilon: float = 5.0                      # ε: privacy budget
    delta: float = 1e-5                       # δ: privacy parameter
    clip_norm_mean: float = 1.0               # S₁: L2 clipping threshold for μ
    clip_norm_cov: float = 5.0                # S₂: L2 clipping threshold for Σ
    psd_epsilon: float = 1e-6                 # ε_PSD: minimum eigenvalue after repair


@dataclass
class RobustnessConfig:
    """Robust aggregation and anomaly detection (Section IV-E)."""
    enable_robust_agg: bool = True            # Toggle Mahalanobis filtering
    tau_percentile: float = 0.999             # χ² percentile for threshold τ
    shrinkage_gamma: float = 0.1              # γ: Ledoit-Wolf shrinkage coefficient
    geometric_median_iters: int = 50          # Weiszfeld iterations for bootstrap
    malicious_fraction: float = 0.0           # Fraction of malicious clients (for attack sim)


@dataclass
class BaselineConfig:
    """Baseline algorithm configuration for all 8 baselines."""
    method: str = "flgss"                     # "flgss", "fedavg", "fedprox", "scaffold", "moon", "fedsam", "fedkd", "fedclustering", "pfedkd", "all"
    fedprox_mu: float = 0.01                  # FedProx proximal coefficient
    scaffold_lr: float = 0.001                # SCAFFOLD server learning rate
    moon_mu: float = 0.1                      # MOON model-contrastive loss weight
    moon_temperature: float = 0.5             # MOON contrastive temperature
    fedsam_rho: float = 0.05                  # FedSAM perturbation neighborhood radius
    fedkd_temperature: float = 2.0            # FedKD distillation temperature
    fedkd_alpha: float = 0.5                  # FedKD distillation loss weight
    fedclustering_num_clusters: int = 3       # FedClustering number of cluster models
    pfedkd_lam: float = 0.5                   # PFedKD personalized interpolation factor
    attack_type: str = "random"               # Byzantine attack type: "random", "label_flip", "scaling"


@dataclass
class FLGSSConfig:
    """Master configuration aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    # Device
    device: str = "cuda"                      # "cuda" or "cpu"

    def __post_init__(self):
        """Apply dataset-specific defaults matching Table 1 & Table 2 in paper."""
        if self.data.dataset == "cifar10":
            self.data.num_classes = 10
            self.model.anchor_type = "resnet18"
            self.model.anchor_output_dim = 512
            self.model.latent_dim = 512
            self.federation.num_clients = 100
            self.federation.clients_per_round = 10
        elif self.data.dataset == "ucihar":
            self.data.num_classes = 6
            self.model.anchor_type = "transformer"
            self.model.transformer_input_dim = 561
            self.model.anchor_output_dim = 128
            self.model.latent_dim = 128
            self.federation.num_clients = 100
            self.federation.clients_per_round = 10
        elif self.data.dataset == "intel":
            self.data.num_classes = 2
            self.model.anchor_type = "transformer"
            self.model.transformer_input_dim = 8
            self.model.anchor_output_dim = 128
            self.model.latent_dim = 128
            self.federation.num_clients = 54
            self.federation.clients_per_round = 5
        elif self.data.dataset == "nbaiot":
            self.data.num_classes = 2
            self.model.anchor_type = "transformer"
            self.model.transformer_input_dim = 115
            self.model.anchor_output_dim = 128
            self.model.latent_dim = 128
            self.federation.num_clients = 500
            self.federation.clients_per_round = 50


def get_cifar10_config(alpha: float = 0.1) -> FLGSSConfig:
    """Convenience factory for CIFAR-10 experiments."""
    cfg = FLGSSConfig()
    cfg.data.dataset = "cifar10"
    cfg.data.alpha = alpha
    cfg.__post_init__()
    return cfg


def get_ucihar_config(alpha: float = 0.1) -> FLGSSConfig:
    """Convenience factory for UCI-HAR experiments."""
    cfg = FLGSSConfig()
    cfg.data.dataset = "ucihar"
    cfg.data.alpha = alpha
    cfg.__post_init__()
    return cfg


def get_intel_config() -> FLGSSConfig:
    """Convenience factory for Intel Berkeley Lab experiments."""
    cfg = FLGSSConfig()
    cfg.data.dataset = "intel"
    cfg.__post_init__()
    return cfg


def get_nbaiot_config() -> FLGSSConfig:
    """Convenience factory for N-BaIoT experiments."""
    cfg = FLGSSConfig()
    cfg.data.dataset = "nbaiot"
    cfg.__post_init__()
    return cfg
