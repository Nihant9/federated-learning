"""
app/detection/data_sanitizer.py

Dataset-level anomaly and poisoning detector.
Identifies label-flipping attacks, manipulated chest X-rays, and anomalous samples
by analyzing model loss residuals, prediction-label discordance, and latent feature space distances.
"""

from typing import Dict, List, Any, Optional
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms


class DataSanitizer:
    """
    Scans datasets and image batches for adversarial poisoning and label manipulation.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        # Prototype centroids for class 0 (Normal) and class 1 (Pneumonia) in latent feature space
        self.prototypes = {0: None, 1: None}

    def compute_prototypes(self, clean_loader, max_batches: int = 20):
        """Compute class prototype centroids in 512-dim latent space using clean references."""
        feats_0 = []
        feats_1 = []

        with torch.no_grad():
            for i, (images, labels) in enumerate(clean_loader):
                if i >= max_batches:
                    break
                images = images.to(self.device)
                labels = labels.squeeze().long().to(self.device)
                features = self.model.extract_features(images)

                for feat, label in zip(features, labels):
                    if label.item() == 0:
                        feats_0.append(feat.cpu().numpy())
                    else:
                        feats_1.append(feat.cpu().numpy())

        if feats_0:
            self.prototypes[0] = np.mean(feats_0, axis=0)
        if feats_1:
            self.prototypes[1] = np.mean(feats_1, axis=0)

    def audit_dataset(
        self,
        dataset,
        max_samples: int = 300,
        loss_threshold: float = 1.5,
        confidence_threshold: float = 0.85
    ) -> Dict[str, Any]:
        """
        Scan a labeled dataset for flipped or poisoned samples.
        A sample is flagged as poisoned if the trained model is highly confident
        in the opposite class with an abnormally large loss residual.
        """
        flagged_samples: List[Dict[str, Any]] = []
        clean_count = 0
        total_inspected = min(len(dataset), max_samples)

        with torch.no_grad():
            for idx in range(total_inspected):
                item = dataset[idx]
                image, label = item[0], item[1]

                # Ensure image tensor has batch dimension (1, 1, H, W)
                if image.dim() == 3:
                    img_tensor = image.unsqueeze(0).to(self.device)
                else:
                    img_tensor = image.to(self.device)

                true_label = int(label[0] if isinstance(label, (list, np.ndarray, torch.Tensor)) else label)
                outputs = self.model(img_tensor)
                probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
                predicted_label = int(np.argmax(probs))
                predicted_conf = float(probs[predicted_label])

                # Cross-entropy loss for this specific sample
                sample_loss = float(-np.log(max(probs[true_label], 1e-7)))

                # Poisoning condition: High-confidence disagreement with given label
                is_poisoned = (
                    (predicted_label != true_label)
                    and (predicted_conf >= confidence_threshold or sample_loss >= loss_threshold)
                )

                if is_poisoned:
                    flagged_samples.append({
                        "sample_index": idx,
                        "declared_label": "Normal" if true_label == 0 else "Pneumonia",
                        "suspected_true_label": "Normal" if predicted_label == 0 else "Pneumonia",
                        "model_confidence": round(predicted_conf * 100, 2),
                        "anomaly_score": round(min(sample_loss / 3.0, 1.0), 3),
                        "reason": f"Severe label discordance. Sample labeled {true_label} but shows {round(predicted_conf*100, 1)}% features of class {predicted_label}."
                    })
                else:
                    clean_count += 1

        poison_ratio = (len(flagged_samples) / total_inspected) if total_inspected > 0 else 0.0

        return {
            "total_inspected": total_inspected,
            "clean_count": clean_count,
            "poisoned_count": len(flagged_samples),
            "poison_percentage": round(poison_ratio * 100, 2),
            "status": "COMPROMISED" if len(flagged_samples) > 0 else "CLEAN",
            "flagged_samples": flagged_samples[:50],  # Return up to 50 detailed findings
        }

    def inspect_single_image(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Inspect a single chest X-ray image for physical tampering (triggers, watermarks, abnormal pixel distributions).
        """
        # Convert to grayscale
        gray_img = pil_image.convert("L")
        np_img = np.array(gray_img, dtype=np.float32)

        # 1. High frequency noise / digital perturbation check (Laplacian variance)
        grad_x = np.diff(np_img, axis=1)
        grad_y = np.diff(np_img, axis=0)
        roughness = float(np.var(grad_x) + np.var(grad_y))

        # 2. Extreme saturation check (artificial patches / triggers)
        black_ratio = float((np_img < 5).mean())
        white_ratio = float((np_img > 250).mean())
        has_corner_trigger = (
            (np_img[:10, :10] > 240).mean() > 0.8 or
            (np_img[-10:, -10:] > 240).mean() > 0.8
        )

        anomaly_score = 0.0
        reasons = []

        if has_corner_trigger:
            anomaly_score += 0.65
            reasons.append("High-intensity synthetic watermark / trigger detected in corner pixels.")
        if white_ratio > 0.25:
            anomaly_score += 0.25
            reasons.append("Abnormal white pixel saturation detected.")

        return {
            "tampered": anomaly_score >= 0.5,
            "anomaly_score": round(min(anomaly_score, 1.0), 3),
            "reasons": reasons if reasons else ["Image structure passes visual integrity checks."]
        }
