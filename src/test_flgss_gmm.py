import torch
import numpy as np

# Simulate high-dimensional semantic vectors from test_cifar_features
from test_cifar_features import feats_tr_224, y_tr_224, feats_te_224, y_te_224

print("\nEvaluating GMM Classifiers on 224x224 ResNet-18 features...")

# Test 1: Diagonal Gaussian / Shrinkage GMM
d = feats_tr_224.size(1)
class_means = {}
class_covs = {}

for c in range(10):
    mask = (y_tr_224 == c)
    z_c = feats_tr_224[mask]
    n_c = z_c.size(0)
    mu_c = z_c.mean(dim=0)
    centered = z_c - mu_c.unsqueeze(0)
    emp_cov = (centered.T @ centered) / n_c
    
    # Adaptive Ledoit-Wolf / shrinkage: cov = (1-gamma)*cov + gamma * avg_var * I
    avg_var = torch.trace(emp_cov) / d
    gamma = 0.5
    shrunk_cov = (1 - gamma) * emp_cov + gamma * avg_var * torch.eye(d) + 1e-4 * torch.eye(d)
    
    class_means[c] = mu_c
    class_covs[c] = shrunk_cov

def classify_gmm(te_z, means, covs):
    B = te_z.size(0)
    log_probs = torch.zeros((B, 10))
    for c in range(10):
        mu = means[c]
        cov = covs[c]
        diff = te_z - mu.unsqueeze(0)
        L = torch.linalg.cholesky(cov)
        v = torch.linalg.solve_triangular(L, diff.T, upper=False)
        mahal = (v ** 2).sum(dim=0)
        log_det = 2 * torch.log(L.diagonal()).sum()
        log_p = -0.5 * (d * np.log(2 * np.pi) + log_det + mahal)
        log_probs[:, c] = log_p
    return log_probs.argmax(dim=1)

preds = classify_gmm(feats_te_224, class_means, class_covs)
acc = (preds == y_te_224).float().mean().item() * 100
print(f"Shrinkage GMM Classification Accuracy on CIFAR-10: {acc:.2f}%")
