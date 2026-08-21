# FLGSS: Federated Generative Semantic Spaces

Official implementation of **Federated Generative Semantic Spaces (FLGSS)** for scalable and trustworthy federated learning across edge/IoT devices.

---

## 🌟 Overview

FLGSS is a framework that transitions federated learning from complex weight-space averaging to stable statistical abstraction in a universal semantic space:
- **Zero Weight Aggregation:** Employs frozen foundation model encoders (ResNet / Transformers) to project local sensory and vision data into a shared semantic latent space.
- **Statistical Abstraction:** Communicates compact class-conditional statistics $(\boldsymbol{\mu}_{k,c}, \boldsymbol{\Sigma}_{k,c})$ instead of large neural network weights.
- **Hierarchical GMM Synthesis:** Edge and Cloud servers synthesize community and global Gaussian Mixture Models for sub-millisecond MAP classification and anomaly detection.
- **Extreme Communication Efficiency:** Reduces communication overhead by $>10\times$ compared to standard FL methods.

---

## 🚀 Key Features & Baselines

Includes a comprehensive benchmark against 8 state-of-the-art and foundational FL baselines:
1. **FedAvg** (McMahan et al., 2017)
2. **FedProx** (Li et al., 2020)
3. **SCAFFOLD** (Karimireddy et al., 2020)
4. **MOON** (Li et al., 2021)
5. **FedSAM** (Qu et al., 2023)
6. **FedKD** (Wu et al., 2022)
7. **FedClustering** (Zhao et al., 2025)
8. **PFedKD** (Li et al., 2025)
9. **FLGSS (Ours)**

Supports 4 standard and real-world non-IID datasets:
- **CIFAR-10** (Vision non-IID Dirichlet $\alpha \in \{10.0, 1.0, 0.1\}$)
- **UCI-HAR** (Human Activity Recognition sensor dataset)
- **Intel Berkeley Lab** (54 real-world IoT sensor nodes)
- **N-BaIoT** (500 IoT botnet detection benchmark)

---

## 📦 Installation

```bash
git clone https://github.com/SuttA-MNNIT/FLGSS.git
cd FLGSS/src
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Run FLGSS on CIFAR-10
```bash
python main.py --dataset cifar10 --alpha 0.1 --rounds 200 --method flgss
```

### 2. Run All 9 Algorithms
```bash
python main.py --dataset cifar10 --alpha 0.1 --rounds 200 --method all --output-dir ./results
```

### 3. Run on Google Colab
Open and run `run_in_colab.ipynb` to execute the experimental campaign on a GPU runtime.

---

## 📁 Repository Structure

```
FLGSS/
├── LICENSE                     # License
├── run_in_colab.ipynb          # Google Colab notebook
└── src/
    ├── baselines/              # Implementations of 8 FL baselines
    ├── client/                 # FLGSS client-side projection & statistics
    ├── data/                   # Dataset loaders and Dirichlet partitioner
    ├── models/                 # Frozen anchor models (ResNet-18, Transformers)
    ├── privacy/                # Local Differential Privacy (LDP)
    ├── server/                 # Edge server aggregation & Cloud synthesis
    ├── trust/                  # Byzantine filtering & anomaly detection
    ├── utils/                  # Metrics tracking & visualization
    ├── config.py               # Hyperparameter configuration
    ├── main.py                 # Main execution entrypoint
    ├── run_full_paper_benchmark.py
    └── requirements.txt
```
