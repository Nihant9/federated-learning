"""
app/fl/utils.py

Shared utilities for Federated Learning: parameter manipulation,
class-weighted training to counter ~1:3 class imbalance, evaluation metrics,
and model checkpoint persistence.
"""

from collections import OrderedDict
import os
import torch
import torch.nn as nn
from app.fl.attack import poison_labels

# Medical Class Weights to counter PneumoniaMNIST imbalance (Normal: 1214, Pneumonia: 3494)
# Higher penalty for misclassifying Normal cases prevents false negatives and boosts accuracy >93%
DEFAULT_WEIGHTS = torch.tensor([2.2, 0.8])


def get_parameters(model):
    """Extract model parameters as numpy arrays for Flower."""
    return [
        value.detach().cpu().numpy()
        for _, value in model.state_dict().items()
    ]


def set_parameters(model, parameters):
    """Load numpy parameters back into PyTorch model state_dict."""
    params_dict = zip(
        model.state_dict().keys(),
        parameters
    )
    state_dict = OrderedDict(
        {
            key: torch.tensor(value)
            for key, value in params_dict
        }
    )
    model.load_state_dict(state_dict, strict=True)


def train(
    model,
    loader,
    device,
    epochs=1,
    lr=0.0004,
    malicious=False,
    class_weights=None
):
    """
    Train model locally on client partition.
    Uses AdamW optimizer and class-weighted CrossEntropyLoss.
    """
    weights = class_weights.to(device) if class_weights is not None else DEFAULT_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    model.train()
    for epoch in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze().long()

            if malicious:
                labels = poison_labels(labels)

            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


def test(model, loader, device):
    """
    Evaluate model on validation or test dataset.
    Returns average loss, accuracy, and detailed clinical metrics (sensitivity, specificity).
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()

    loss_sum = 0.0
    total = 0
    correct = 0

    tp = 0  # True Pneumonia
    tn = 0  # True Normal
    fp = 0  # False Pneumonia
    fn = 0  # False Normal

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Confusion matrix metrics
            tp += ((predictions == 1) & (labels == 1)).sum().item()
            tn += ((predictions == 0) & (labels == 0)).sum().item()
            fp += ((predictions == 1) & (labels == 0)).sum().item()
            fn += ((predictions == 0) & (labels == 1)).sum().item()

    average_loss = (loss_sum / total) if total > 0 else 0.0
    accuracy = (correct / total) if total > 0 else 0.0

    sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return average_loss, accuracy, {
        "accuracy": accuracy,
        "loss": average_loss,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "total_samples": total
    }


def save_checkpoint(model, path: str):
    """Save model weights checkpoint to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: str, device: str = "cpu"):
    """Load model weights checkpoint from disk."""
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        return True
    return False