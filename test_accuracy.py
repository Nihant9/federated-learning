"""
test_accuracy.py

Standalone manual evaluation script to test the accuracy, sensitivity,
specificity, and confusion matrix of the trained Federated Global Model.

Usage:
    python test_accuracy.py
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# Add backend to Python module path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

from app.fl.model import PneumoniaCNN
from app.data.loader import load_datasets
from app.fl.utils import test, load_checkpoint


def evaluate_model_manually():
    print("=" * 65)
    print("   MEDSHIELD AI — FEDERATED MODEL ACCURACY & CLINICAL EVALUATION")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device : {device}")

    # Checkpoint path
    checkpoint_path = os.path.join(BACKEND_DIR, "app", "fl", "global_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"[!] Error: Model checkpoint not found at: {checkpoint_path}")
        return

    print(f"[*] Checkpoint Path : {checkpoint_path}")
    print("[*] Loading PneumoniaMNIST test dataset (64x64 resolution)...")

    # Load dataset
    _, test_dataset = load_datasets(image_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model and load weights
    model = PneumoniaCNN().to(device)
    success = load_checkpoint(model, checkpoint_path, device=device)
    if not success:
        print("[!] Failed to load model weights.")
        return

    print("[*] Running inference on test dataset...")

    # Evaluate
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    loss_sum = 0.0
    total = 0
    correct = 0
    tp = 0  # True Pneumonia
    tn = 0  # True Normal
    fp = 0  # False Pneumonia (Predict Pneumonia, Actually Normal)
    fn = 0  # False Normal (Predict Normal, Actually Pneumonia)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            tp += ((predictions == 1) & (labels == 1)).sum().item()
            tn += ((predictions == 0) & (labels == 0)).sum().item()
            fp += ((predictions == 1) & (labels == 0)).sum().item()
            fn += ((predictions == 0) & (labels == 1)).sum().item()

    avg_loss = loss_sum / total if total > 0 else 0.0
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    f1_score = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    print("\n" + "-" * 65)
    print("                    EVALUATION RESULTS")
    print("-" * 65)
    print(f"  * Total Test Samples      : {total} chest radiographs")
    print(f"  * Overall Test Accuracy   : {accuracy:.2f}% ({correct}/{total} correct)")
    print(f"  * Test Loss               : {avg_loss:.4f}")
    print(f"  * Sensitivity / Recall    : {sensitivity:.2f}% (Pneumonia detection rate)")
    print(f"  * Specificity (TNR)       : {specificity:.2f}% (Normal lung identification)")
    print(f"  * Precision (PPV)         : {precision:.2f}%")
    print(f"  * F1 Score                : {f1_score:.2f}%")
    print("-" * 65)

    print("\n" + "-" * 65)
    print("                     CONFUSION MATRIX")
    print("-" * 65)
    print(f"  True Positives  (TP - Correct Pneumonia) : {tp:>4} / {tp + fn} ({tp/(tp+fn)*100:.1f}%)")
    print(f"  True Negatives  (TN - Correct Normal)    : {tn:>4} / {tn + fp} ({tn/(tn+fp)*100:.1f}%)")
    print(f"  False Positives (FP - Normal as Pneumonia): {fp:>4} / {tn + fp}")
    print(f"  False Negatives (FN - Missed Pneumonia)  : {fn:>4} / {tp + fn}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    evaluate_model_manually()
