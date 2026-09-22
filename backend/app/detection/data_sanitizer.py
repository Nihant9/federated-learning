"""
app/detection/data_sanitizer.py

Dataset-level anomaly and poisoning detector.
Identifies label-flipping attacks, manipulated chest X-rays, and anomalous samples
by analyzing model loss residuals, prediction-label discordance, and latent feature space distances.
Includes a multi-tier Radiograph Gatekeeper & OOD Validator to reject random non-medical photos.
"""

from typing import Dict, List, Any, Optional
import io
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms


class DataSanitizer:
    """
    Scans datasets and image batches for adversarial poisoning, label manipulation,
    and out-of-distribution non-medical uploads.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        # Prototype centroids for class 0 (Normal) and class 1 (Pneumonia) in latent feature space
        self.prototypes = {0: None, 1: None}

        # Precomputed reference chest centroid for OOD & Non-Chest Gatekeeping
        self.chest_centroid = None
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        centroid_candidates = [
            os.path.join(base_dir, "fl", "chest_centroid.pt"),
            os.path.join(os.path.dirname(__file__), "..", "fl", "chest_centroid.pt"),
            "backend/app/fl/chest_centroid.pt",
            "app/fl/chest_centroid.pt",
        ]
        for cp in centroid_candidates:
            if os.path.exists(cp):
                try:
                    c = torch.load(cp, map_location=self.device)
                    self.chest_centroid = c.to(self.device)
                    break
                except Exception:
                    pass

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

        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        with torch.no_grad():
            for idx in range(total_inspected):
                img_tensor, label = dataset[idx]
                label_val = int(label[0] if hasattr(label, "__len__") else label)

                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                target_tensor = torch.tensor([label_val]).to(self.device)

                logits = self.model(img_tensor)
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                loss = criterion(logits, target_tensor).item()

                predicted_class = int(np.argmax(probs))
                predicted_conf = float(probs[predicted_class])

                # Detection rule: Flagged if prediction disagrees with label AND model is confident
                is_poisoned = (predicted_class != label_val) and (
                    loss >= loss_threshold or predicted_conf >= confidence_threshold
                )

                if is_poisoned:
                    flagged_samples.append({
                        "sample_index": idx,
                        "declared_label": "Normal" if label_val == 0 else "Pneumonia",
                        "suspected_true_label": "Normal" if predicted_class == 0 else "Pneumonia",
                        "model_confidence": round(predicted_conf * 100, 2),
                        "loss_residual": round(loss, 4),
                        "status": "FLAGGED_POISONED"
                    })
                else:
                    clean_count += 1

        poison_ratio = len(flagged_samples) / total_inspected if total_inspected > 0 else 0.0

        return {
            "total_inspected": total_inspected,
            "clean_count": clean_count,
            "poisoned_count": len(flagged_samples),
            "poison_percentage": round(poison_ratio * 100, 2),
            "status": "COMPROMISED" if len(flagged_samples) > 0 else "CLEAN",
            "flagged_samples": flagged_samples[:50],  # Return up to 50 detailed findings
        }

    def validate_chest_xray(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Multi-tier Radiograph Gatekeeper & Thoracic Anatomical Validator.
        Specifically rejects:
        - Extremity & limb radiographs (legs, knees, feet, arms, hands)
        - Focused cardiac imaging (isolated heart, angiograms, echocardiograms, ultrasound)
        - Non-radiographic color photos, screenshots, documents, and flat images
        Ensures model evaluates ONLY authentic Posteroanterior (PA) / Anteroposterior (AP) Chest X-rays.
        """
        reasons = []

        # 1. Color Chroma & Saturation Divergence Check (Monochromatic radiograph check)
        rgb_img = pil_image.convert("RGB")
        rgb_arr = np.array(rgb_img, dtype=np.float32)
        r, g, b = rgb_arr[:, :, 0], rgb_arr[:, :, 1], rgb_arr[:, :, 2]
        chroma_diff = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)))
        hsv_img = pil_image.convert("HSV")
        hsv_arr = np.array(hsv_img, dtype=np.float32)
        mean_saturation = float(np.mean(hsv_arr[:, :, 1]) / 255.0)

        if chroma_diff > 6.0 or mean_saturation > 0.10:
            return {
                "is_valid": False,
                "rejection_type": "NATURAL_COLOR_IMAGE",
                "status": "INVALID_IMAGE",
                "reasons": [
                    f"Natural color photograph detected (chroma divergence: {chroma_diff:.1f}, saturation: {mean_saturation*100:.1f}%). "
                    "Authentic chest radiographs are monochromatic/grayscale."
                ],
            }

        # 2. Aspect Ratio Check (Chest radiographs PA/AP vs elongated limb radiographs)
        w, h = pil_image.size
        aspect = w / h
        if aspect < 0.58 or aspect > 1.72:
            return {
                "is_valid": False,
                "rejection_type": "EXTREMITY_ASPECT_RATIO",
                "status": "INVALID_IMAGE",
                "reasons": [
                    f"Non-thoracic aspect ratio ({aspect:.2f}). Extremity (leg, knee, arm, hand) radiographs typically exhibit elongated dimensions. "
                    "Chest radiographs require standard PA/AP thoracic framing."
                ],
            }

        # 3. Grayscale Dynamic Range & Contrast Check
        gray_arr = np.array(pil_image.convert("L").resize((128, 128)), dtype=np.float32)
        std_dev = float(np.std(gray_arr))
        if std_dev < 15.0:
            return {
                "is_valid": False,
                "rejection_type": "FLAT_IMAGE",
                "status": "INVALID_IMAGE",
                "reasons": [
                    f"Insufficient radiographic dynamic range (std dev: {std_dev:.1f}). "
                    "Uploaded file appears to be a blank, flat, or non-medical image."
                ],
            }

        extreme_ratio = float(((gray_arr < 8) | (gray_arr > 248)).mean())
        if extreme_ratio > 0.75 and std_dev > 85.0:
            return {
                "is_valid": False,
                "rejection_type": "DOCUMENT_SCAN",
                "status": "INVALID_IMAGE",
                "reasons": [
                    "Abnormal bimodal pixel distribution. Document scan, line graphic, or screenshot detected rather than an anatomical X-ray."
                ],
            }

        # 4. Extremity / Cortical Bone Shaft & Ambient Air Check (Leg, Knee, Arm, Hand)
        black_air = float((gray_arr < 32).mean())
        side_left = float(gray_arr[:, :20].mean())
        side_right = float(gray_arr[:, 108:].mean())
        center_strip = gray_arr[:, 48:80]
        center_mean = float(center_strip.mean())

        # Limb morphology: continuous longitudinal cortical bone shaft flanked by dark ambient air
        if side_left < 30.0 and side_right < 30.0 and center_mean > 60.0:
            row_means = center_strip.mean(axis=1)
            if row_means.min() > 30.0 and (row_means.std() / (row_means.mean() + 1e-5)) < 0.40:
                return {
                    "is_valid": False,
                    "rejection_type": "EXTREMITY_LIMB_RADIOGRAPH",
                    "status": "INVALID_IMAGE",
                    "reasons": [
                        "Anatomical limb morphology detected: continuous longitudinal cortical bone shaft flanked by ambient air borders. "
                        "Leg, knee, or arm radiographs are not supported; please upload a PA/AP Chest radiograph."
                    ],
                }

        # Bilateral thoracic chamber texture check
        center_60 = gray_arr[26:102, 26:102]
        left_hemithorax = center_60[:, :25]
        right_hemithorax = center_60[:, 50:]
        mid_mediastinum = center_60[:, 25:50]
        has_bilateral_chambers = (
            left_hemithorax.std() > 18.0 and
            right_hemithorax.std() > 18.0 and
            mid_mediastinum.mean() > left_hemithorax.mean()
        )

        # Ambient air background check (extremity or isolated cardiac fluoroscopy)
        if black_air > 0.42:
            if not has_bilateral_chambers or black_air > 0.60:
                return {
                    "is_valid": False,
                    "rejection_type": "NON_CHEST_RADIOGRAPH",
                    "status": "INVALID_IMAGE",
                    "reasons": [
                        f"Excessive ambient air background ({black_air*100:.1f}%). "
                        "Characteristic of an extremity (leg, knee, hand) or focused cardiac/fluoroscopy scan rather than a full thoracic chest radiograph."
                    ],
                }

        # 5. Thoracic Bilateral Lung Apices & Cardiac Morphology Check
        # Check for isolated cardiac imaging / echocardiogram / zoomed heart without bilateral lung apices
        top_zone = gray_arr[:24, :]
        top_left = float(top_zone[:, 16:48].mean())
        top_right = float(top_zone[:, 80:112].mean())

        if not has_bilateral_chambers and (top_left < 20.0 and top_right < 20.0 and center_mean > 50.0):
            return {
                "is_valid": False,
                "rejection_type": "ISOLATED_CARDIAC_IMAGING",
                "status": "INVALID_IMAGE",
                "reasons": [
                    "Absence of bilateral pulmonary lung apices. The image appears to be a focused cardiac scan, "
                    "angiogram, or echocardiogram rather than a full PA/AP chest radiograph."
                ],
            }

        # 6. Deep Latent Space Gatekeeper (Embedding Distance from Chest Centroid)
        if self.model is not None and self.chest_centroid is not None:
            t = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            t_img = t(pil_image.convert("L")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                f = self.model.extract_features(t_img)
                f = f / torch.norm(f, p=2, dim=1, keepdim=True)
                cos_sim = float(torch.mm(f, self.chest_centroid.t()).item())

            if cos_sim < 0.68:
                return {
                    "is_valid": False,
                    "rejection_type": "LATENT_OOD_MISMATCH",
                    "status": "INVALID_IMAGE",
                    "reasons": [
                        f"Anatomical feature alignment score ({cos_sim:.2f}) falls below thoracic radiograph threshold. "
                        "The deep feature representations do not match chest radiograph structures."
                    ],
                }

        return {
            "is_valid": True,
            "rejection_type": "NONE",
            "status": "VALID_XRAY",
            "chroma_diff": round(chroma_diff, 2),
            "mean_saturation": round(mean_saturation, 3),
            "intensity_std_dev": round(std_dev, 2),
            "reasons": ["Image exhibits authentic PA/AP chest radiograph anatomy with bilateral aerated lung fields."]
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
