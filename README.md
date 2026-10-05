# FLGSS v2.0: Federated Generative Semantic Spaces

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Accelerated-NVIDIA%20H100%20%2F%20A100-76b900.svg)](https://www.nvidia.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **Federated Generative Semantic Spaces (FLGSS v2.0)** for scalable, robust, and communication-efficient federated intelligence across edge and IoT networks.

---

## 🌟 Key Highlights & Innovations

1. **Semantic-Space Aggregation vs. Weight Averaging:**  
   Unlike FedAvg/FedProx which average diverging neural network weights in parameter space, FLGSS extracts class-conditional Gaussian statistics $(\boldsymbol{\mu}_{k,c}, \boldsymbol{\Sigma}_{k,c})$ in a shared latent space defined by a frozen Anchor Model ($E_A$).
2. **On-Device Calibration & Anchor Offloading:**  
   Clients calibrate a tiny local Adapter ($A_{\phi_k}$) to match the Anchor representations during an initial setup phase. In active deployment, IoT devices offload the heavy anchor model and rely strictly on the lightweight adapter, saving up to **40% energy** and reducing peak memory by over **3x**.
3. **Ledoit-Wolf Covariance Regularization:**  
   Guarantees numerical stability and positive definiteness across non-IID partitions with small local sample sizes.
4. **Autonomous Byzantine Robustness:**  
   Edge servers compute an aggregate Mahalanobis anomaly score $S_k$ against current community distributions (with Geometric Median bootstrapping at round $t=1$). Rejects poisoning attacks (e.g. 20% Byzantine attackers) with zero manual oversight.
5. **Client-Level Differential Privacy (CL-DP):**  
   Implements $L_2$ norm clipping, calibrated Gaussian perturbation, and post-processing eigenvalue reconstruction to strictly maintain the positive semi-definite (PSD) nature of covariance matrices.
6. **Architectural Fairness:**  
   All compared baselines (FedAvg, FedProx, SCAFFOLD, MOON, FedSAM, FedKD) utilize the **exact same frozen ResNet-18 backbone** as FLGSS, training downstream heads/adapters to ensure strict architectural fairness (Fair Backbone Protocol).

---

## 📁 Repository Structure

```
FLGSS-3-10-26/
├── FLGSS_H100_Colab.ipynb         # Interactive, cell-by-cell runnable Google Colab notebook
├── README.md                      # Complete system documentation and guides
├── requirements.txt               # Dependencies (PyTorch, Torchvision, Scikit-learn, etc.)
├── train.py                       # Modular CLI runner for single experiments
├── run_experiments.py             # Benchmark orchestrator (Tables I, II, III, IV & Plots)
├── flgss/                         # Core algorithmic package
│   ├── config.py                  # Dataclass with all experiment and hardware hyperparams
│   ├── flgss_core.py              # FLGSS on-device distillation and server orchestration
│   ├── aggregation.py             # Global GMM class-conditional distributions & MAP inference
│   ├── defense.py                 # Mahalanobis anomaly filtering & Byzantine attack simulator
│   ├── privacy.py                 # Client-Level Differential Privacy & PSD projection
│   ├── models/                    # Neural architectures
│   │   ├── anchor.py              # Frozen ResNet-18, ResNet-50, and Sensor Encoders
│   │   ├── adapter.py             # Lightweight local Adapter Networks (A_{\phi_k})
│   │   ├── baselines_models.py    # Downstream heads for fair baseline comparisons
│   │   └── autoencoder.py         # Federated Autoencoder (FedAE) for anomaly detection
│   ├── baselines/                 # SOTA Federated Learning Algorithms
│   │   ├── fedavg.py              # FedAvg (McMahan et al. 2017)
│   │   ├── fedprox.py             # FedProx with proximal regularization (Li et al. 2020)
│   │   ├── scaffold.py            # SCAFFOLD with control variates (Karimireddy et al. 2020)
│   │   ├── moon.py                # MOON model-contrastive learning (Li et al. 2021)
│   │   ├── fedsam.py              # FedSAM sharpness-aware minimization (Qu et al. 2023)
│   │   └── fedkd.py               # FedKD / PFedKD mutual knowledge distillation
│   ├── datasets/                  # Heterogeneous benchmark loaders
│   │   ├── partition.py           # Dirichlet Non-IID skew partitioner (alpha = 0.1, 1.0, 10.0)
│   │   ├── cifar10.py             # CIFAR-10 vision dataset
│   │   ├── uci_har.py             # UCI Human Activity Recognition (561 features, 6 classes)
│   │   ├── intel_lab.py           # Intel Berkeley Lab environmental sensor motes
│   │   └── nbaiot.py              # N-BaIoT botnet attack detection (115 features, 11 classes)
│   └── utils/                     # Metrics, profiling, and plotting
│       ├── logger.py              # CSV metrics logger
│       ├── metrics.py             # Accuracy, F1, and Anomaly Detection AUC
│       ├── profiling.py           # Communication cost, peak GPU memory & energy models
│       └── plotting.py            # High-legibility publication-ready figure generator
```

---

## 🚀 Quick Start on Google Colab (H100 / A100)

1. **Upload the Folder:**  
   Zip and upload the repository directory to your Google Drive or directly to your Colab session.
2. **Select GPU Accelerator:**  
   Navigate to `Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ select **NVIDIA H100** or **A100**.
3. **Open the Notebook:**  
   Open `FLGSS_H100_Colab.ipynb` and run the cells sequentially.

---

## 💻 CLI Usage (`train.py`)

### 1. Run FLGSS on CIFAR-10 (High Non-IID Skew $\alpha=0.1$)
```bash
python train.py --dataset cifar10 --algo flgss --alpha 0.1 --rounds 100 --clients 100 --C 0.1
```

### 2. Run Baselines under Fair Backbone Setup
```bash
# FedAvg
python train.py --dataset cifar10 --algo fedavg --alpha 0.1 --rounds 100

# FedProx
python train.py --dataset cifar10 --algo fedprox --alpha 0.1 --mu 0.01 --rounds 100

# SCAFFOLD
python train.py --dataset cifar10 --algo scaffold --alpha 0.1 --rounds 100

# FedKD
python train.py --dataset cifar10 --algo fedkd --alpha 0.1 --rounds 100
```

### 3. Run Sensor Dataset (UCI-HAR)
```bash
python train.py --dataset uci_har --algo flgss --alpha 0.1 --rounds 100
```

### 4. Byzantine Robustness & Model Poisoning Attack
```bash
# 20% Byzantine Attack WITHOUT Mahalanobis filter (Collapses to ~43%)
python train.py --dataset cifar10 --algo flgss --alpha 0.1 --attack gaussian_shift --attacker_ratio 0.2 --no_robust

# 20% Byzantine Attack WITH Mahalanobis filter (Maintains ~82.5%)
python train.py --dataset cifar10 --algo flgss --alpha 0.1 --attack gaussian_shift --attacker_ratio 0.2 --robust
```

### 5. Client-Level Differential Privacy (CL-DP)
```bash
python train.py --dataset cifar10 --algo flgss --alpha 0.1 --enable_dp --dp_epsilon 2.0
```

---

## 📊 Full Benchmark Suite (`run_experiments.py`)

To reproduce the benchmark tables and figures:

```bash
# Run Table I (Non-IID Accuracy Comparison across all algorithms)
python run_experiments.py --suite table1 --fast

# Run Table II (Anomaly Detection AUC Score vs. FedAE)
python run_experiments.py --suite table2

# Run Table IV & Ablation Studies (Under Attack, No Anchor, etc.)
python run_experiments.py --suite table4

# Generate All Benchmark Figures (comm_cost.pdf, abl1.pdf, abl2.pdf, abl3.pdf)
python run_experiments.py --suite plots
```

---

## 📈 Benchmark Reference Results

### Table I: Final Test Accuracy (%) vs. Degree of Non-IID Skew ($\alpha$)
| Algorithm | Core Paradigm | CIFAR-10 ($\alpha=0.1$) | UCI-HAR ($\alpha=0.1$) | Intel Dataset | N-BaIoT |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **FedAvg** | Weight Averaging | 68.95 ± 1.21% | 85.27 ± 1.15% | 81.44 ± 1.42% | 86.41 ± 1.2% |
| **FedProx** | Reg. Weight Avg. | 69.58 ± 1.10% | 86.05 ± 1.05% | 82.17 ± 1.35% | 87.12 ± 1.1% |
| **SCAFFOLD** | Corrected W. Avg. | 69.31 ± 1.15% | 85.88 ± 1.10% | 81.95 ± 1.38% | 86.95 ± 1.2% |
| **MOON** | Repr. Contrastive | 72.84 ± 0.98% | 89.05 ± 0.90% | 85.19 ± 1.20% | 90.54 ± 0.9% |
| **FedSAM** | Sharpness Aware | 73.15 ± 0.92% | 89.44 ± 0.85% | 86.45 ± 1.15% | 91.20 ± 0.8% |
| **FedKD** | Distillation | 74.90 ± 0.90% | 89.80 ± 0.82% | 87.10 ± 1.05% | 92.40 ± 0.8% |
| **FedClustering** | Semantic Clustering | 76.88 ± 0.80% | 90.15 ± 0.75% | 88.65 ± 0.95% | 93.10 ± 0.7% |
| **PFedKD** | Personalized KD | 76.50 ± 0.85% | 90.30 ± 0.78% | 88.40 ± 0.95% | 93.55 ± 0.6% |
| **FLGSS (Ours)** | **Semantic GMM** | **85.23 ± 0.42%** | **93.52 ± 0.40%** | **95.61 ± 0.55%** | **98.15 ± 0.2%** |

### Table II: Unsupervised Anomaly Detection (AUC Score)
| Algorithm | CIFAR-10 AUC | Intel Berkeley Lab AUC |
| :--- | :---: | :---: |
| **Federated Autoencoder (FedAE)** | 0.79 | 0.82 |
| **FLGSS (Ours)** | **0.98** | **0.99** |

### Table III: On-Device Resource Usage per Client Round
| Task | Algorithm | Peak Memory (MB) | Energy (Joules) | Comm. Overhead |
| :--- | :--- | :---: | :---: | :---: |
| **Vision (CIFAR-10)** | FedAvg / FedProx | 285.4 MB | 18.2 J | 9,146 - 10,171 MB |
| | **FLGSS (Ours)** | **89.2 MB** | **11.5 J** | **830 MB (>11x Lower)** |
| **Sensor (HAR / Intel)** | FedAvg / FedProx | 112.5 MB | 9.8 J | Moderate |
| | **FLGSS (Ours)** | **35.1 MB** | **5.1 J** | **>10x Lower** |

### Benchmark Figures
- **Convergence Curves (`Figure_1.pdf` / `Figure_4_Updated.pdf`)**: Convergence curves comparing all 9 algorithms on non-IID CIFAR-10 ($\alpha=0.1$).
- **Communication Cost (`comm_cost.pdf`)**: Horizontal lollipop plot on log scale showing total communication cost (MB) to reach 70% accuracy.
- **Anchor Model Sensitivity (`abl1.pdf`)**: Anchor Model capacity sensitivity (Scratch vs ResNet-18 vs ResNet-50).
- **Latent Dimension Trade-off (`abl2.pdf`)**: Semantic space latent dimension trade-off ($d \in \{32, 64, 128, 256, 512\}$).
- **Differential Privacy Trade-off (`abl3.pdf`)**: Client-level Differential Privacy utility-privacy trade-off ($\epsilon \in \{1.0, 2.0, 5.0, 10.0, 20.0\}$).

---

## 🛡️ License
This project is open-source under the MIT License.
