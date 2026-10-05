import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_data(data_path: str = "./data", batch_size: int = 32, test_batch_size: int = 512, num_workers: int = 4, pin_memory: bool = True):
    """
    Downloads and loads CIFAR-10 with standard preprocessing.
    """
    os.makedirs(data_path, exist_ok=True)

    # ImageNet native resolution and normalization for pre-trained ResNet-18 Anchor
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Clean dataset without random crop for consistent anchor embedding caching
    transform_clean = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_clean)
    test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_dataset, test_loader, test_dataset
