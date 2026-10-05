import os
import torch
import dataclasses
from typing import List, Optional

@dataclasses.dataclass
class FLConfig:
    # --- Experiment Settings ---
    dataset: str = "cifar10"              # ['cifar10', 'uci_har', 'intel_lab', 'nbaiot']
    algo: str = "flgss"                   # ['flgss', 'fedavg', 'fedprox', 'scaffold', 'moon', 'fedsam', 'fedkd']
    exp_name: Optional[str] = None
    seed: int = 42

    # --- Federated Network Parameters ---
    num_clients: int = 100               # N total clients (Benchmark configuration)
    client_fraction: float = 0.1         # C = 0.1 -> 10 clients selected per round
    num_rounds: int = 100                # Total communication rounds (e.g. 100 or 200)
    local_epochs: int = 5                # E on-device local training epochs
    batch_size: int = 32                 # Local batch size
    test_batch_size: int = 512           # Global evaluation batch size
    learning_rate: float = 0.001         # Client local learning rate (Adam or SGD)
    momentum: float = 0.9                # SGD momentum for baselines
    weight_decay: float = 1e-4

    # --- Non-IID Skew ---
    alpha: float = 0.1                   # Dirichlet parameter (0.1 = High Skew, 1.0 = Medium, 10.0 = Low)

    # --- Model & Representation Settings ---
    latent_dim: int = 512                # Latent dimension d (512 for ResNet-18, 128 for sensor)
    fair_backbone: bool = True           # Use frozen Anchor Backbone across ALL baselines (Fair Backbone Protocol)
    backbone_type: str = "resnet18"      # ['resnet18', 'resnet50', 'vit', 'swin_t', 'transformer', 'mlp']
    pretrained: bool = True              # Use pre-trained weights (Ablation flag)
    adapter_hidden_dim: int = 512        # Hidden dim of local Adapter MLP
    num_classes: int = 10                # Updated automatically based on dataset

    # --- FLGSS Specific Parameters ---
    gmm_shrinkage: float = 0.1           # Ledoit-Wolf shrinkage gamma (Eq. 5)
    gmm_reg_covar: float = 1e-5          # Regularization added to diagonal for PSD stability
    cache_anchor: bool = True            # Precompute anchor representations for 10x-50x H100 simulation speedup
    global_sync_freq: int = 1            # Frequency of cloud server synchronization

    # --- Trustworthiness & Defense (Section III-C) ---
    robust_filtering: bool = False       # Enable Mahalanobis anomaly filtering (tested in Section IV-D)
    robust_threshold: float = 10.0       # Tau threshold for rejection (normalized Mahalanobis distance)
    attack_type: str = "none"            # ['none', 'label_flip', 'noise', 'gaussian_shift']
    attacker_ratio: float = 0.0          # Fraction of malicious clients (e.g., 0.1 or 0.2 in Ablation)
    attack_noise_scale: float = 5.0      # Scale for Byzantine parameter shift

    # --- Privacy Parameters (Section III-D) ---
    enable_dp: bool = False              # Enable Client-Level Differential Privacy
    dp_epsilon: float = 2.0              # Privacy budget epsilon
    dp_delta: float = 1e-5               # Privacy budget delta
    dp_clip_mean: float = 10.0           # S1 clipping threshold for mean norm
    dp_clip_cov: float = 20.0            # S2 clipping threshold for covariance norm
    dp_sigma: float = 1.0                # Calibrated noise multiplier
    eps_psd: float = 1e-4                # Minimum eigenvalue threshold after noise addition

    # --- Baseline Specific Parameters ---
    mu: float = 0.01                     # FedProx proximal regularization weight
    moon_mu: float = 1.0                 # MOON contrastive loss weight
    moon_temp: float = 0.5               # MOON temperature
    sam_rho: float = 0.05                # FedSAM perturbation radius
    kd_alpha: float = 0.5                # FedKD distillation weight
    kd_temperature: float = 2.0          # FedKD temperature

    # --- Hardware & Colab Optimization (H100 / A100 / RTX) ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True                 # Automatic Mixed Precision (FP16 / BF16)
    use_tf32: bool = True                # Enable TF32 for matrix multiplications
    num_workers: int = 4
    pin_memory: bool = True
    data_path: str = "./data"
    output_dir: str = "./runs"

    def __post_init__(self):
        if self.dataset == "cifar10":
            self.num_classes = 10
            self.latent_dim = 512
        elif self.dataset == "uci_har":
            self.num_classes = 6
            self.latent_dim = 128
        elif self.dataset == "intel_lab":
            self.num_classes = 4
            self.latent_dim = 64
        elif self.dataset == "nbaiot":
            self.num_classes = 11
            self.latent_dim = 128
