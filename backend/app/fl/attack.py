"""
app/fl/attack.py
Poisoning attack logic — kept separate from utils.py so the generic
training code has no knowledge of attacks at all. A malicious client
calls poison_labels() before training; an honest client never touches
this file.
"""

import torch


def poison_labels(labels: torch.Tensor) -> torch.Tensor:
    """
    Label-flipping attack for a binary classification task
    (PneumoniaMNIST: 0 = Normal, 1 = Pneumonia).

    Flips every label to its opposite class, so a malicious client's
    local model learns the wrong association between an X-ray and its
    diagnosis — and its resulting update actively pulls the shared
    global model in the wrong direction.
    """
    return 1 - labels
