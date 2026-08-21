import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

# Load pre-trained ResNet-18
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

# Remove classification head
encoder = torch.nn.Sequential(*list(model.children())[:-1])
encoder.eval()

# Load CIFAR-10 test set with standard normalization
transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_32 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset_224 = torchvision.datasets.CIFAR10(root='./data_cache', train=True, download=True, transform=transform_224)
testset_224 = torchvision.datasets.CIFAR10(root='./data_cache', train=False, download=True, transform=transform_224)

trainset_32 = torchvision.datasets.CIFAR10(root='./data_cache', train=True, download=True, transform=transform_32)
testset_32 = torchvision.datasets.CIFAR10(root='./data_cache', train=False, download=True, transform=transform_32)

print("Testing 32x32 vs 224x224 feature extraction with pre-trained ResNet-18...")

def extract(dataset, num=2000):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    feats, labels = [], []
    count = 0
    with torch.no_grad():
        for x, y in loader:
            f = encoder(x).squeeze()
            if f.dim() == 1:
                f = f.unsqueeze(0)
            feats.append(f)
            labels.append(y)
            count += len(y)
            if count >= num:
                break
    return torch.cat(feats)[:num], torch.cat(labels)[:num]

feats_tr_32, y_tr_32 = extract(trainset_32, 2000)
feats_te_32, y_te_32 = extract(testset_32, 1000)

feats_tr_224, y_tr_224 = extract(trainset_224, 2000)
feats_te_224, y_te_224 = extract(testset_224, 1000)

def eval_ncm(tr_f, tr_y, te_f, te_y, name=""):
    means = []
    for c in range(10):
        c_mask = (tr_y == c)
        means.append(tr_f[c_mask].mean(dim=0))
    means = torch.stack(means) # (10, 512)

    # Cosine similarity
    norm_means = means / (means.norm(dim=1, keepdim=True) + 1e-8)
    norm_te = te_f / (te_f.norm(dim=1, keepdim=True) + 1e-8)
    preds_cos = (norm_te @ norm_means.T).argmax(dim=1)
    acc_cos = (preds_cos == te_y).float().mean().item() * 100

    # Euclidean
    dists = torch.cdist(te_f, means)
    preds_euc = dists.argmin(dim=1)
    acc_euc = (preds_euc == te_y).float().mean().item() * 100

    print(f"[{name}] Nearest Class Mean -> Cosine Acc: {acc_cos:.2f}%, Euclidean Acc: {acc_euc:.2f}%")

eval_ncm(feats_tr_32, y_tr_32, feats_te_32, y_te_32, "32x32 Raw CIFAR")
eval_ncm(feats_tr_224, y_tr_224, feats_te_224, y_te_224, "224x224 Resized CIFAR")
