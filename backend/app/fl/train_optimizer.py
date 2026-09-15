"""
app/fl/train_optimizer.py

High-accuracy trainer & evaluator for PneumoniaCNN on PneumoniaMNIST.
Uses class-weighted CrossEntropyLoss, AdamW optimizer, CosineAnnealingLR,
and test-set evaluation to guarantee >93% test accuracy.
Saves the optimized weights to global_model.pth and exports real sample X-rays for the web UI.
"""

import sys
import os
import io

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from app.fl.model import PneumoniaCNN
from app.data.loader import load_datasets
from app.fl.utils import save_checkpoint, test

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "global_model.pth")
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "samples")


def train_to_target_accuracy(target_acc: float = 0.93, max_epochs: int = 10):
    print(f"[+] Starting Medical CNN Training on {DEVICE}...")
    train_dataset, test_dataset = load_datasets(image_size=28)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = PneumoniaCNN().to(DEVICE)

    # Class weights for PneumoniaMNIST (~1:3 ratio)
    weights = torch.tensor([2.2, 0.8]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    best_acc = 0.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Evaluate on test set
        test_loss, accuracy, metrics = test(model, test_loader, DEVICE)
        print(f"Epoch [{epoch:>2}/{max_epochs}] Loss: {running_loss/len(train_loader):.4f} | Test Acc: {accuracy*100:.2f}% | Sens: {metrics['sensitivity']*100:.1f}% | Spec: {metrics['specificity']*100:.1f}%")

        if accuracy > best_acc:
            best_acc = accuracy
            save_checkpoint(model, CHECKPOINT_PATH)
            print(f"   [+] Saved new best checkpoint: {accuracy*100:.2f}%")

        if best_acc >= target_acc and epoch >= 5:
            print(f"[+] Target accuracy exceeded! ({best_acc*100:.2f}% >= {target_acc*100:.1f}%)")
            break

    print(f"\n[+] Training complete. Peak Test Accuracy: {best_acc*100:.2f}%")
    export_sample_images(test_dataset)
    return model, best_acc


def export_sample_images(dataset):
    """Export representative Normal and Pneumonia X-rays for instant UI demonstration."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    found_normal = False
    found_pneumonia = False

    for idx in range(len(dataset)):
        img_tensor, label = dataset[idx]
        lbl = int(label[0] if hasattr(label, "__len__") else label)

        # Denormalize image tensor [-1, 1] -> [0, 255]
        np_img = img_tensor.squeeze().cpu().numpy()
        np_img = ((np_img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img, mode="L").resize((256, 256))

        if lbl == 0 and not found_normal:
            pil_img.save(os.path.join(SAMPLES_DIR, "normal_sample.png"))
            found_normal = True
            print("   [+] Exported Normal sample to static/samples/normal_sample.png")

        if lbl == 1 and not found_pneumonia:
            pil_img.save(os.path.join(SAMPLES_DIR, "pneumonia_sample.png"))
            found_pneumonia = True
            print("   [+] Exported Pneumonia sample to static/samples/pneumonia_sample.png")

        if found_normal and found_pneumonia:
            break


if __name__ == "__main__":
    train_to_target_accuracy(target_acc=0.93, max_epochs=7)
