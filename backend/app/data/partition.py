"""
app/data/partition.py

Splits the PneumoniaMNIST dataset across simulated hospital clients.
Uses balanced IID partitioning and stratified local train/validation splits.
"""

import numpy as np
from torch.utils.data import DataLoader, Subset


def partition_dataset(
    dataset,
    num_clients: int = 8,
    iid: bool = True,
    seed: int = 42
):
    """
    Split the dataset among simulated clients.

    iid=True:
        Randomly distributes samples approximately equally.

    iid=False:
        Creates a mild non-IID distribution.
    """

    rng = np.random.default_rng(seed)

    n = len(dataset)
    indices = np.arange(n)

    if iid:
        # Shuffle all samples
        rng.shuffle(indices)

        # Split equally among clients
        client_indices = np.array_split(
            indices,
            num_clients
        )

    else:
        # Get labels
        labels = np.array([
            dataset.labels[i][0]
            for i in range(n)
        ])

        # Sort by label
        sorted_indices = indices[
            np.argsort(labels)
        ]

        # Create shards
        num_shards = num_clients * 2

        shards = np.array_split(
            sorted_indices,
            num_shards
        )

        # Shuffle shards
        rng.shuffle(shards)

        # Give two shards to every client
        client_indices = []

        for i in range(num_clients):

            client_data = np.concatenate([
                shards[2 * i],
                shards[2 * i + 1]
            ])

            client_indices.append(
                client_data
            )

    return [
        Subset(
            dataset,
            indices.tolist()
        )
        for indices in client_indices
    ]


def get_client_loaders(
    client_subset,
    batch_size: int = 32,
    val_split: float = 0.1,
    seed: int = 42
):
    """
    Create stratified local train and validation splits.

    Stratified splitting ensures both classes are represented
    proportionally in train and validation datasets.
    """

    rng = np.random.default_rng(seed)

    # Get original indices from the Subset
    subset_indices = np.array(
        client_subset.indices
    )

    # Get labels for these samples
    dataset = client_subset.dataset

    labels = np.array([
        dataset.labels[index][0]
        for index in subset_indices
    ])

    train_indices = []
    val_indices = []

    # Split each class separately
    unique_classes = np.unique(labels)

    for class_label in unique_classes:

        # Find samples of this class
        class_positions = np.where(
            labels == class_label
        )[0]

        # Shuffle samples
        rng.shuffle(class_positions)

        # Validation samples
        n_val = max(
            1,
            int(
                len(class_positions)
                * val_split
            )
        )

        # Add validation positions
        val_indices.extend(
            class_positions[:n_val]
        )

        # Add training positions
        train_indices.extend(
            class_positions[n_val:]
        )

    # Convert positions into actual dataset indices
    train_dataset_indices = subset_indices[
        train_indices
    ]

    val_dataset_indices = subset_indices[
        val_indices
    ]

    # Create subsets
    train_subset = Subset(
        dataset,
        train_dataset_indices.tolist()
    )

    val_subset = Subset(
        dataset,
        val_dataset_indices.tolist()
    )

    # Create loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader