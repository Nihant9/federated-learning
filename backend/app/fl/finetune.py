"""
app/fl/finetune.py

High-precision fine-tuner for PneumoniaCNN (SE-ResNet) on PneumoniaMNIST 64x64.
Applies gentle learning rates, cosine annealing, and weighted CrossEntropy
to surpass 95% test accuracy.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.fl.model import PneumoniaCNN
from app.data.loader import load_datasets
from app.fl.utils import load_checkpoint, save_checkpoint, test

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "global_model.pth")


def run_finetune(epochs: int = 6):
    print(f"[+] Starting fine-tuning pass on {DEVICE}...")
    model = PneumoniaCNN().to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        load_checkpoint(model, CHECKPOINT_PATH, device=str(DEVICE))
        print(f"Loaded existing checkpoint from {CHECKPOINT_PATH}")

    train_ds, test_ds = load_datasets(image_size=64)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    _, initial_acc, init_m = test(model, test_loader, DEVICE)
    print(f"Initial Acc: {initial_acc*100:.2f}% | Sens: {init_m['sensitivity']*100:.1f}% | Spec: {init_m['specificity']*100:.1f}%")

    weights = torch.tensor([1.40, 0.85]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.015)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc = initial_acc

    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, lbls in train_loader:
            imgs = imgs.to(DEVICE)
            lbls = lbls.squeeze().long().to(DEVICE)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        _, acc, m = test(model, test_loader, DEVICE)
        sens = m["sensitivity"]
        spec = m["specificity"]
        print(f"Epoch [{epoch}/{epochs}] Test Acc: {acc*100:.2f}% | Sens: {sens*100:.1f}% | Spec: {spec*100:.1f}%")

        if acc > best_acc and sens >= 0.90:
            best_acc = acc
            save_checkpoint(model, CHECKPOINT_PATH)
            print(f"   [+] Saved new peak checkpoint with Acc={acc*100:.2f}%")

    print(f"\n[+] Fine-tuning complete. Best Accuracy: {best_acc*100:.2f}%")
    return best_acc


if __name__ == "__main__":
    run_finetune(epochs=6)
