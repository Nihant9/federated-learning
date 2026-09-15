"""
app/data/loader.py

Downloads and preprocesses the PneumoniaMNIST medical imaging dataset.
Applies medical data augmentation to improve generalization and robust classification.
"""

import torchvision.transforms as transforms


def get_transforms(image_size: int = 28):
    """Return train and test data transformations."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
        transforms.RandomHorizontalFlip(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    return train_transform, test_transform


def load_datasets(image_size: int = 28):
    """
    Load train and test splits for PneumoniaMNIST.
    Downloads dataset on first call if not present.
    """
    from medmnist import PneumoniaMNIST

    train_transform, test_transform = get_transforms(image_size)

    train_dataset = PneumoniaMNIST(
        split="train",
        transform=train_transform,
        download=True
    )

    test_dataset = PneumoniaMNIST(
        split="test",
        transform=test_transform,
        download=True
    )

    return train_dataset, test_dataset