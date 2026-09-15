"""
app/fl/utils.py

Shared helpers for Federated Learning.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from app.fl.attack import poison_labels


def get_parameters(model):

    return [
        value.detach().cpu().numpy()
        for _, value in model.state_dict().items()
    ]


def set_parameters(model, parameters):

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

    model.load_state_dict(
        state_dict,
        strict=True
    )


def train(
    model,
    loader,
    device,
    epochs=1,
    lr=0.0005,
    malicious=False
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        for images, labels in loader:

            images = images.to(device)

            labels = labels.squeeze().long()

            if malicious:
                labels = poison_labels(labels)

            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item()


def test(model, loader, device):

    criterion = nn.CrossEntropyLoss()

    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.squeeze().long()

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss_sum += (
                loss.item()
                * labels.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


    average_loss = (
        loss_sum / total
        if total > 0
        else 0.0
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return average_loss, accuracy