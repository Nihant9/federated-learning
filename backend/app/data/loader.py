import torchvision.transforms as transforms


def load_datasets():

    from medmnist import PneumoniaMNIST

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        ),
    ])

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