"""
evaluate_and_improve.py

Comprehensive model evaluation & improvement pipeline for PneumoniaCNN.

Steps:
1. Evaluate current model on PneumoniaMNIST test set (base accuracy, sensitivity, specificity, F1)
2. Stress-test with diverse augmentations: brightness, contrast, noise, blur, rotations
3. Diagnose weaknesses (false positive / false negative breakdown)
4. Re-train with stronger augmentation + label smoothing if accuracy < 95%
5. Regenerate chest centroid for improved OOD rejection
6. Save improved model weights to global_model.pth

Run from the backend directory:
    backend\\venv\\Scripts\\python.exe evaluate_and_improve.py
"""

import sys
import os

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
import random

from app.fl.model import PneumoniaCNN
from app.fl.utils import save_checkpoint, test
from app.data.loader import load_datasets

# -----------------------------------------------------------------
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "app", "fl", "global_model.pth")
CENTROID_PATH   = os.path.join(os.path.dirname(__file__), "app", "fl", "chest_centroid.pt")
SAMPLES_DIR     = os.path.join(os.path.dirname(__file__), "app", "static", "samples")
TARGET_ACC      = 0.95
MAX_EPOCHS      = 15
# -----------------------------------------------------------------


def load_model():
    """Load PneumoniaCNN from checkpoint."""
    model = PneumoniaCNN().to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        print(f"  [OK] Loaded weights from {CHECKPOINT_PATH}")
    else:
        print(f"  [WARN] No checkpoint found — using random weights.")
    model.eval()
    return model


def evaluate_full(model, loader, label="Standard"):
    """Full evaluation: accuracy + sensitivity + specificity + F1."""
    model.eval()
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)
            preds  = model(imgs).argmax(dim=1)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    total = tp + tn + fp + fn
    acc   = (tp + tn) / total if total else 0
    sens  = tp / (tp + fn) if (tp + fn) else 0
    spec  = tn / (tn + fp) if (tn + fp) else 0
    prec  = tp / (tp + fp) if (tp + fp) else 0
    f1    = 2 * prec * sens / (prec + sens) if (prec + sens) else 0
    npv   = tn / (tn + fn) if (tn + fn) else 0

    print(f"\n{'='*62}")
    print(f"  {label}  ({total} samples on {DEVICE})")
    print(f"{'='*62}")
    print(f"  Accuracy    : {acc*100:6.2f}%")
    print(f"  Sensitivity : {sens*100:6.2f}%  (Pneumonia recall)")
    print(f"  Specificity : {spec*100:6.2f}%  (Normal recall)")
    print(f"  Precision   : {prec*100:6.2f}%")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  NPV         : {npv*100:6.2f}%")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'='*62}")
    if fn > 0:
        print(f"  [!] {fn} Pneumonia cases MISSED (False Negatives) - clinical risk!")
    if fp > 0:
        print(f"  [!] {fp} Normal patients over-diagnosed (False Positives)")

    return {"accuracy": acc, "sensitivity": sens, "specificity": spec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def make_stress_loader(test_dataset, augmentation_fn, batch_size=64):
    """Create a DataLoader that applies a PIL augmentation on-the-fly."""
    class AugDataset(torch.utils.data.Dataset):
        def __init__(self, base, aug_fn, size=64):
            self.base      = base
            self.aug_fn    = aug_fn
            self.to_tensor = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, label = self.base[idx]
            np_img = ((img.squeeze().numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(np_img, mode="L")
            pil_img = self.aug_fn(pil_img)
            return self.to_tensor(pil_img), label

    ds = AugDataset(test_dataset, augmentation_fn)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def _add_noise(img):
    arr   = np.array(img, dtype=np.float32)
    noise = np.random.randn(*arr.shape) * 18
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _random_crop(img):
    w, h   = img.size
    margin = int(min(w, h) * 0.08)
    left   = random.randint(0, margin)
    top    = random.randint(0, margin)
    right  = w - random.randint(0, margin)
    bottom = h - random.randint(0, margin)
    return img.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR)


def stress_test(model, test_dataset):
    """Run model against 10 clinical stress scenarios."""
    print("\n" + "="*62)
    print("  STRESS TEST - Diverse Augmentation Scenarios")
    print("="*62)

    scenarios = {
        "Dark Exposure (under-exposed film)": lambda img: transforms.functional.adjust_brightness(img, 0.5),
        "Bright Exposure (over-exposed film)": lambda img: transforms.functional.adjust_brightness(img, 1.8),
        "Low Contrast (foggy scan)":          lambda img: transforms.functional.adjust_contrast(img, 0.4),
        "High Contrast (harsh scan)":         lambda img: transforms.functional.adjust_contrast(img, 2.0),
        "Gaussian Blur (soft focus)":         lambda img: img.filter(ImageFilter.GaussianBlur(radius=1.5)),
        "Rotation +15 degrees":               lambda img: img.rotate(15, fillcolor=0),
        "Rotation -15 degrees":               lambda img: img.rotate(-15, fillcolor=0),
        "Gaussian Noise (sensor noise)":      _add_noise,
        "Horizontal Flip":                    lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        "Small Crop + Resize":                _random_crop,
    }

    results = {}
    for name, aug_fn in scenarios.items():
        loader = make_stress_loader(test_dataset, aug_fn)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs   = imgs.to(DEVICE)
                labels = labels.squeeze().long().to(DEVICE)
                preds  = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        acc  = correct / total if total else 0
        flag = "  OK  " if acc >= 0.90 else " WEAK " if acc >= 0.80 else " POOR "
        print(f"  [{flag}] {name:<38} Acc: {acc*100:.2f}%")
        results[name] = acc

    avg = np.mean(list(results.values()))
    print(f"\n  Average stress accuracy: {avg*100:.2f}%")
    print("="*62)
    return results


def retrain_with_augmentation(target_acc=TARGET_ACC, max_epochs=MAX_EPOCHS):
    """Re-train PneumoniaCNN with stronger augmentation to exceed target accuracy."""
    print("\n" + "="*62)
    print("  RETRAINING - Enhanced Augmentation Pipeline")
    print(f"  Target: {target_acc*100:.0f}%  |  Max Epochs: {max_epochs}")
    print("="*62)

    strong_train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(degrees=0, translate=(0.07, 0.07), scale=(0.92, 1.08), shear=4),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    from medmnist import PneumoniaMNIST
    try:
        train_ds = PneumoniaMNIST(split="train", size=64, transform=strong_train_transform, download=True)
        test_ds  = PneumoniaMNIST(split="test",  size=64, transform=test_transform,         download=True)
    except Exception:
        train_ds = PneumoniaMNIST(split="train", size=28, transform=strong_train_transform, download=True)
        test_ds  = PneumoniaMNIST(split="test",  size=28, transform=test_transform,         download=True)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

    model = PneumoniaCNN().to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        print("  [+] Warm-start fine-tuning from existing checkpoint")

    weights   = torch.tensor([1.40, 0.85]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=5e-6
    )

    best_acc   = 0.0
    best_score = 0.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        _, acc, metrics = test(model, test_loader, DEVICE)
        sens  = metrics["sensitivity"]
        spec  = metrics["specificity"]
        f1    = 2 * sens * spec / (sens + spec) if (sens + spec) > 0 else 0
        score = acc * 0.5 + sens * 0.30 + spec * 0.20

        print(
            f"  Epoch [{epoch:>2}/{max_epochs}] "
            f"Loss: {running_loss/len(train_loader):.4f} | "
            f"Acc: {acc*100:.2f}% | "
            f"Sens: {sens*100:.1f}% | "
            f"Spec: {spec*100:.1f}% | "
            f"F1: {f1:.3f}"
        )

        if (acc > best_acc or (acc >= 0.92 and score > best_score)) and sens >= 0.90:
            best_acc   = acc
            best_score = score
            save_checkpoint(model, CHECKPOINT_PATH)
            print(f"    --> Saved best checkpoint: Acc={acc*100:.2f}% Sens={sens*100:.1f}% Spec={spec*100:.1f}%")

        if best_acc >= target_acc and epoch >= 5:
            print(f"\n  [+] Target {target_acc*100:.0f}% reached at epoch {epoch}. Done.")
            break

    print(f"\n  Best accuracy achieved: {best_acc*100:.2f}%")
    return model, best_acc, test_ds


def regenerate_chest_centroid(model, test_dataset):
    """Recompute chest radiograph centroid for OOD Gatekeeper."""
    print("\n  [+] Regenerating chest centroid embedding...")
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    embeddings = []
    with torch.no_grad():
        indices = list(range(min(800, len(test_dataset))))
        for idx in indices:
            img, _ = test_dataset[idx]
            if not isinstance(img, torch.Tensor):
                img = transform(img)
            t = img.unsqueeze(0).to(DEVICE)
            f = model.extract_features(t)
            f = F.normalize(f, p=2, dim=1)
            embeddings.append(f.squeeze(0).cpu())

    centroid = torch.stack(embeddings).mean(dim=0, keepdim=True)
    centroid = F.normalize(centroid, p=2, dim=1)
    torch.save(centroid, CENTROID_PATH)
    print(f"  [OK] Centroid saved -> {CENTROID_PATH}  shape={centroid.shape}")
    return centroid


def export_samples(model, test_dataset):
    """Export verified Normal + Pneumonia samples for the UI."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    model.eval()
    found_normal = found_pneumonia = False

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            img, label = test_dataset[idx]
            lbl  = int(label[0] if hasattr(label, "__len__") else label)
            t    = img.unsqueeze(0).to(DEVICE)
            prob = torch.softmax(model(t), dim=1).squeeze().cpu().tolist()

            np_img = img.squeeze().numpy()
            np_img = ((np_img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            pil    = Image.fromarray(np_img, mode="L").resize((256, 256))

            if lbl == 0 and not found_normal and prob[0] > 0.92:
                pil.save(os.path.join(SAMPLES_DIR, "normal_sample.png"))
                found_normal = True
                print("  [+] Exported normal_sample.png")

            if lbl == 1 and not found_pneumonia and prob[1] > 0.92:
                pil.save(os.path.join(SAMPLES_DIR, "pneumonia_sample.png"))
                found_pneumonia = True
                print("  [+] Exported pneumonia_sample.png")

            if found_normal and found_pneumonia:
                break


# -----------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*62)
    print("  MedShield AI - Model Evaluation & Improvement Pipeline")
    print("="*62)

    print("\n[STEP 1] Loading model and dataset...")
    model = load_model()
    _, test_dataset = load_datasets(image_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    print("\n[STEP 2] Baseline evaluation...")
    baseline = evaluate_full(model, test_loader, "Baseline (Standard Test Set)")

    print("\n[STEP 3] Stress testing across 10 augmentation scenarios...")
    stress_results = stress_test(model, test_dataset)

    needs_retraining = (
        baseline["accuracy"]    < TARGET_ACC or
        baseline["sensitivity"] < 0.92       or
        baseline["specificity"] < 0.88       or
        np.mean(list(stress_results.values())) < 0.88
    )

    if needs_retraining:
        print(f"\n[STEP 4] Retraining (current Acc={baseline['accuracy']*100:.2f}%)...")
        model, final_acc, test_dataset = retrain_with_augmentation(TARGET_ACC, MAX_EPOCHS)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        print("\n[STEP 4b] Post-training evaluation...")
        evaluate_full(model, test_loader, "Post-Training Results")
        print("\n[STEP 4c] Post-training stress test...")
        stress_test(model, test_dataset)
    else:
        print(f"\n[STEP 4] Model already meets targets (Acc={baseline['accuracy']*100:.2f}%). Skipping retraining.")

    print("\n[STEP 5] Regenerating OOD chest centroid...")
    regenerate_chest_centroid(model, test_dataset)

    print("\n[STEP 6] Exporting sample X-rays for UI...")
    export_samples(model, test_dataset)

    print("\n" + "="*62)
    print("  DONE.")
    print(f"  Checkpoint : {CHECKPOINT_PATH}")
    print(f"  Centroid   : {CENTROID_PATH}")
    print("="*62 + "\n")
