"""
app/data/loader.py

Downloads and preprocesses the PneumoniaMNIST medical imaging dataset.
Applies medical data augmentation to improve generalization and robust classification.
Supports 64x64 or 28x28 resolution to preserve fine lung parenchyma and infiltrates.
"""

import torchvision.transforms as transforms


def get_transforms(image_size: int = 64):
    """Return train and test data transformations with medical data augmentation."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=8),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    return train_transform, test_transform


def load_datasets(image_size: int = 64):
    """
    Load train and test splits for PneumoniaMNIST.
    Downloads dataset on first call if not present.
    Supports size=64 or size=28.
    """
    from medmnist import PneumoniaMNIST

    train_transform, test_transform = get_transforms(image_size)

    try:
        train_dataset = PneumoniaMNIST(
            split="train",
            size=image_size,
            transform=train_transform,
            download=True
        )
        test_dataset = PneumoniaMNIST(
            split="test",
            size=image_size,
            transform=test_transform,
            download=True
        )
    except Exception:
        # Fallback to 28x28 if 64x64 fails
        train_dataset = PneumoniaMNIST(
            split="train",
            size=28,
            transform=train_transform,
            download=True
        )
        test_dataset = PneumoniaMNIST(
            split="test",
            size=28,
            transform=test_transform,
            download=True
        )

    return train_dataset, test_dataset