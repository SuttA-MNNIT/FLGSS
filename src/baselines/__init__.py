"""
Federated Learning Baseline Algorithms
======================================
1. FedAvg (McMahan et al., 2017)
2. FedProx (Li et al., 2020)
3. SCAFFOLD (Karimireddy et al., 2020)
4. MOON (Li et al., 2021)
5. FedSAM (Qu et al., 2023)
6. FedKD (Wu et al., 2022)
7. FedClustering (Zhao et al., 2025)
8. PFedKD (Li et al., 2025)
"""

from baselines.fedavg import BaselineClassifier, FedAvgClient, FedAvgServer
from baselines.fedprox import FedProxClient, FedProxServer
from baselines.scaffold import SCAFFOLDClient, SCAFFOLDServer
from baselines.moon import MOONClient, MOONServer
from baselines.fedsam import FedSAMClient, FedSAMServer
from baselines.fedkd import FedKDClient, FedKDServer
from baselines.fedclustering import FedClusteringClient, FedClusteringServer
from baselines.pfedkd import PFedKDClient, PFedKDServer

__all__ = [
    "BaselineClassifier",
    "FedAvgClient", "FedAvgServer",
    "FedProxClient", "FedProxServer",
    "SCAFFOLDClient", "SCAFFOLDServer",
    "MOONClient", "MOONServer",
    "FedSAMClient", "FedSAMServer",
    "FedKDClient", "FedKDServer",
    "FedClusteringClient", "FedClusteringServer",
    "PFedKDClient", "PFedKDServer",
]
