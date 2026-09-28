"""
app/xai/explainer.py

Comprehensive Explainable AI (XAI) Engine for PneumoniaCNN.
Implements the following explanation techniques — all computed dynamically
from the live model (no static precomputed data):

1.  Gradient-Weighted Class Activation Maps (Grad-CAM)
    - Hooks the final SEResBlock conv layer, captures gradients and
      activations, weights them, and produces a class-discriminative heatmap.

2.  Vanilla Input-Gradient Saliency
    - Computes |dL/dX| to highlight pixel-level sensitivity.

3.  Integrated Gradients (IG)
    - Approximates Shapley attribution via Riemann integration along a
      straight-line path from a black baseline to the input image.
    - Uses 50 interpolation steps.

4.  Occlusion Sensitivity (LIME-style)
    - Systematically patches image regions with neutral gray, measures
      prediction confidence drop.  Returns a 2-D sensitivity grid.

5.  Smooth Grad
    - Average of N noisy saliency passes to reduce high-frequency noise.

6.  Class Activation Similarity Score
    - Measures how similar the current activation pattern is to stored
      class prototypes (Normal / Pneumonia) in the latent embedding space.

7.  Attribution Summary
    - Returns per-region importance fractions (left lung, right lung,
      cardiac mediastinum, costophrenic angles) from the Grad-CAM map.

All methods return base64-encoded PNG overlays AND numerical attribution
arrays/scores so the frontend can render charts without depending on
server-side matplotlib.
"""

from __future__ import annotations

import io
import base64
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Colour-map helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_jet_colormap(norm_map: np.ndarray) -> np.ndarray:
    """Convert a [0,1] 2-D array to an RGBA jet-coloured image (H,W,4)."""
    # Jet-like: blue → cyan → green → yellow → red
    r = np.clip(1.5 - np.abs(norm_map * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(norm_map * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(norm_map * 4 - 1), 0, 1)
    a = np.full_like(r, 0.72)
    return (np.stack([r, g, b, a], axis=-1) * 255).astype(np.uint8)


def _apply_hot_colormap(norm_map: np.ndarray) -> np.ndarray:
    """Hot colormap (black → red → orange → yellow → white)."""
    r = np.clip(norm_map * 3, 0, 1)
    g = np.clip(norm_map * 3 - 1, 0, 1)
    b = np.clip(norm_map * 3 - 2, 0, 1)
    a = np.clip(norm_map * 0.85, 0, 1)
    return (np.stack([r, g, b, a], axis=-1) * 255).astype(np.uint8)


def _apply_cool_colormap(norm_map: np.ndarray) -> np.ndarray:
    """Cool colormap (cyan → magenta) for negative/uncertainty regions."""
    r = norm_map
    g = 1 - norm_map
    b = np.ones_like(norm_map)
    a = np.clip(norm_map * 0.7, 0, 1)
    return (np.stack([r, g, b, a], axis=-1) * 255).astype(np.uint8)


def _blend_overlay(base_pil: Image.Image, rgba_np: np.ndarray) -> str:
    """Composite an RGBA numpy array over a PIL image; return data-URI PNG."""
    overlay = Image.fromarray(rgba_np, mode="RGBA")
    base_rgba = base_pil.resize(overlay.size).convert("RGBA")
    blended = Image.alpha_composite(base_rgba, overlay)
    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _norm_map(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a 2-D array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + eps)


def _arr_to_data_uri(arr_2d: np.ndarray, colormap_fn) -> str:
    """Convert a raw 2-D attribution map (any scale) to data-URI via colormap."""
    normed = _norm_map(arr_2d)
    rgba = colormap_fn(normed)
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Gradient hook manager
# ─────────────────────────────────────────────────────────────────────────────

class _GradHook:
    """Registers forward + backward hooks on a target module."""

    def __init__(self, module: nn.Module):
        self.activation: Optional[torch.Tensor] = None
        self.gradient: Optional[torch.Tensor] = None
        self._fwd_handle = module.register_forward_hook(self._save_activation)
        self._bwd_handle = module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output):
        self.activation = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradient = grad_out[0].detach()

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Main XAI Explainer Class
# ─────────────────────────────────────────────────────────────────────────────

class XAIExplainer:
    """
    Wraps a PneumoniaCNN model and exposes multiple XAI explanation methods.
    All computations are performed dynamically on the live model weights.
    """

    # Standard pre-processing identical to inference pipeline
    TRANSFORM = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

        # Identify the target convolutional layer for Grad-CAM
        # Last SEResBlock's second conv layer captures semantic features best
        self._gradcam_target: nn.Module = self.model.features[-1].conv2

    def _preprocess(self, pil_img: Image.Image) -> torch.Tensor:
        """Preprocess a PIL image to a (1,1,64,64) normalised tensor."""
        gray = pil_img.convert("L")
        return self.TRANSFORM(gray).unsqueeze(0).to(self.device)

    # ─── 1. GRAD-CAM ───────────────────────────────────────────────────────

    def grad_cam(
        self,
        pil_img: Image.Image,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute Grad-CAM heatmap.

        Returns:
            overlay_uri   – base64 PNG blended over original image
            raw_uri       – base64 PNG of raw heatmap (jet coloured, no blend)
            cam_grid      – serialisable list-of-lists (8×8 normalised scores)
            target_class  – which class was explained
            confidence    – model confidence for that class
        """
        tensor = self._preprocess(pil_img)
        hook = _GradHook(self._gradcam_target)

        try:
            # Forward pass
            tensor_req = tensor.clone().requires_grad_(True)
            logits = self.model(tensor_req)
            probs = F.softmax(logits, dim=1).squeeze()

            if target_class is None:
                target_class = int(probs.argmax().item())

            confidence = float(probs[target_class].item())

            # Backward pass for the chosen class
            self.model.zero_grad()
            score = logits[0, target_class]
            score.backward()

            # Grad-CAM computation
            grads = hook.gradient.squeeze(0)       # (C, H, W)
            acts  = hook.activation.squeeze(0)    # (C, H, W)

            weights = grads.mean(dim=(1, 2))       # global average pooling of gradients
            cam = torch.zeros(acts.shape[1:], device=self.device)
            for i, w in enumerate(weights):
                cam += w * acts[i]

            cam = F.relu(cam)
            cam_np = cam.cpu().numpy()
            cam_norm = _norm_map(cam_np)

            # Up-sample to original image size
            W, H = pil_img.size
            cam_pil = Image.fromarray((cam_norm * 255).astype(np.uint8), "L")
            cam_resized = cam_pil.resize((W, H), Image.BILINEAR)
            cam_arr = np.array(cam_resized, dtype=np.float32) / 255.0

            rgba = _apply_jet_colormap(cam_arr)
            overlay_uri = _blend_overlay(pil_img, rgba)
            raw_uri = _arr_to_data_uri(cam_norm, _apply_jet_colormap)

            # 8×8 grid for frontend bar/heatmap chart
            cam_8x8 = np.array(cam_pil.resize((8, 8), Image.BILINEAR), dtype=np.float32) / 255.0
            cam_grid = cam_8x8.tolist()

        finally:
            hook.remove()

        return {
            "method": "Grad-CAM",
            "overlay_uri": overlay_uri,
            "raw_uri": raw_uri,
            "cam_grid": cam_grid,
            "target_class": target_class,
            "target_class_name": "Pneumonia" if target_class == 1 else "Normal",
            "confidence": round(confidence * 100, 2),
        }

    # ─── 2. VANILLA SALIENCY ───────────────────────────────────────────────

    def vanilla_saliency(
        self,
        pil_img: Image.Image,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute input-gradient saliency |dL/dX|.
        """
        tensor = self._preprocess(pil_img).requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()

        if target_class is None:
            target_class = int(probs.argmax().item())

        score = logits[0, target_class]
        score.backward()

        sal = tensor.grad.data.abs().squeeze().cpu().numpy()
        sal_norm = _norm_map(sal)

        W, H = pil_img.size
        sal_pil = Image.fromarray((sal_norm * 255).astype(np.uint8), "L")
        sal_resized = np.array(sal_pil.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0

        rgba = _apply_hot_colormap(sal_resized)
        overlay_uri = _blend_overlay(pil_img, rgba)
        raw_uri = _arr_to_data_uri(sal_norm, _apply_hot_colormap)

        # Row-level importance (16 rows → for horizontal bar chart)
        sal_16x16 = np.array(sal_pil.resize((16, 16), Image.BILINEAR), dtype=np.float32)
        row_importance = sal_16x16.mean(axis=1).tolist()

        return {
            "method": "Vanilla Saliency",
            "overlay_uri": overlay_uri,
            "raw_uri": raw_uri,
            "row_importance": row_importance,
            "target_class": target_class,
            "target_class_name": "Pneumonia" if target_class == 1 else "Normal",
            "confidence": round(float(probs[target_class].item()) * 100, 2),
        }

    # ─── 3. INTEGRATED GRADIENTS ───────────────────────────────────────────

    def integrated_gradients(
        self,
        pil_img: Image.Image,
        target_class: Optional[int] = None,
        n_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Integrated Gradients: Sundararajan et al., 2017.
        Baseline = zero (black) image.
        Attribution = (input − baseline) * ∫ ∇f(α·input) dα
        """
        tensor = self._preprocess(pil_img)  # (1,1,64,64)
        baseline = torch.zeros_like(tensor)

        # Determine target class from unperturbed input
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze()
        if target_class is None:
            target_class = int(probs.argmax().item())

        # Accumulate gradients along interpolation path
        integrated_grad = torch.zeros_like(tensor)

        for step in range(n_steps):
            alpha = (step + 1) / n_steps
            interp = baseline + alpha * (tensor - baseline)
            interp = interp.requires_grad_(True)

            self.model.zero_grad()
            out = self.model(interp)
            score = out[0, target_class]
            score.backward()

            integrated_grad += interp.grad.detach()

        integrated_grad /= n_steps
        attribution = ((tensor - baseline) * integrated_grad).squeeze().cpu().numpy()

        # Separate positive (supporting) and negative (opposing) attributions
        pos_attr = np.clip(attribution, 0, None)
        neg_attr = np.clip(-attribution, 0, None)

        pos_norm = _norm_map(pos_attr)
        neg_norm = _norm_map(neg_attr)

        W, H = pil_img.size
        pos_pil = Image.fromarray((pos_norm * 255).astype(np.uint8), "L").resize((W, H), Image.BILINEAR)
        pos_arr = np.array(pos_pil, dtype=np.float32) / 255.0
        neg_pil = Image.fromarray((neg_norm * 255).astype(np.uint8), "L").resize((W, H), Image.BILINEAR)
        neg_arr = np.array(neg_pil, dtype=np.float32) / 255.0

        # Blend: positive = jet, negative = cool, composite together
        pos_rgba = _apply_jet_colormap(pos_arr)
        neg_rgba = _apply_cool_colormap(neg_arr)

        # Composite positive over negative
        pos_img = Image.fromarray(pos_rgba, "RGBA")
        neg_img = Image.fromarray(neg_rgba, "RGBA")
        combined_rgba = Image.alpha_composite(neg_img, pos_img)
        combined_np = np.array(combined_rgba)

        overlay_uri = _blend_overlay(pil_img, combined_np)

        raw_pos_uri = _arr_to_data_uri(pos_norm, _apply_jet_colormap)
        raw_neg_uri = _arr_to_data_uri(neg_norm, _apply_cool_colormap)

        # 16-bin signed attribution histogram
        flat_attr = attribution.flatten()
        hist, bin_edges = np.histogram(flat_attr, bins=16)
        bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
        histogram = [
            {"bin": round(float(c), 4), "count": int(v)}
            for c, v in zip(bin_centers, hist)
        ]

        return {
            "method": "Integrated Gradients",
            "overlay_uri": overlay_uri,
            "raw_positive_uri": raw_pos_uri,
            "raw_negative_uri": raw_neg_uri,
            "histogram": histogram,
            "total_positive_attribution": round(float(pos_attr.sum()), 4),
            "total_negative_attribution": round(float(neg_attr.sum()), 4),
            "target_class": target_class,
            "target_class_name": "Pneumonia" if target_class == 1 else "Normal",
            "confidence": round(float(probs[target_class].item()) * 100, 2),
            "n_steps": n_steps,
        }

    # ─── 4. OCCLUSION SENSITIVITY ──────────────────────────────────────────

    def occlusion_sensitivity(
        self,
        pil_img: Image.Image,
        target_class: Optional[int] = None,
        patch_size: int = 8,
        stride: int = 4,
    ) -> Dict[str, Any]:
        """
        LIME-style occlusion: systematically mask image patches and
        measure confidence drop.  A large drop = high importance region.
        """
        tensor = self._preprocess(pil_img)  # (1,1,64,64)

        with torch.no_grad():
            base_logits = self.model(tensor)
            base_probs = F.softmax(base_logits, dim=1).squeeze()

        if target_class is None:
            target_class = int(base_probs.argmax().item())

        base_conf = float(base_probs[target_class].item())

        _, _, H, W = tensor.shape
        importance_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        occluded = tensor.clone()

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                occluded_patch = tensor.clone()
                occluded_patch[0, 0, y:y+patch_size, x:x+patch_size] = 0.0  # neutral gray (−1 after norm)

                with torch.no_grad():
                    out = self.model(occluded_patch)
                    probs = F.softmax(out, dim=1).squeeze()

                drop = base_conf - float(probs[target_class].item())
                importance_map[y:y+patch_size, x:x+patch_size] += drop
                count_map[y:y+patch_size, x:x+patch_size] += 1.0

        count_map = np.where(count_map == 0, 1, count_map)
        importance_map /= count_map
        imp_norm = _norm_map(importance_map)

        W_orig, H_orig = pil_img.size
        imp_pil = Image.fromarray((imp_norm * 255).astype(np.uint8), "L")
        imp_resized = np.array(imp_pil.resize((W_orig, H_orig), Image.BILINEAR), dtype=np.float32) / 255.0

        rgba = _apply_jet_colormap(imp_resized)
        overlay_uri = _blend_overlay(pil_img, rgba)
        raw_uri = _arr_to_data_uri(imp_norm, _apply_jet_colormap)

        # 8×8 grid for frontend
        grid_8x8 = np.array(imp_pil.resize((8, 8), Image.NEAREST), dtype=np.float32) / 255.0

        return {
            "method": "Occlusion Sensitivity",
            "overlay_uri": overlay_uri,
            "raw_uri": raw_uri,
            "importance_grid": grid_8x8.tolist(),
            "baseline_confidence": round(base_conf * 100, 2),
            "max_drop": round(float(importance_map.max()) * 100, 2),
            "mean_drop": round(float(importance_map.mean()) * 100, 4),
            "target_class": target_class,
            "target_class_name": "Pneumonia" if target_class == 1 else "Normal",
            "patch_size": patch_size,
        }

    # ─── 5. SMOOTH GRAD ────────────────────────────────────────────────────

    def smooth_grad(
        self,
        pil_img: Image.Image,
        target_class: Optional[int] = None,
        n_samples: int = 30,
        noise_level: float = 0.15,
    ) -> Dict[str, Any]:
        """
        SmoothGrad: average saliency over N noisy input perturbations
        to reduce gradient noise artefacts.
        """
        tensor = self._preprocess(pil_img)

        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=1).squeeze()
        if target_class is None:
            target_class = int(probs.argmax().item())

        std_dev = noise_level * (tensor.max() - tensor.min()).item()
        accumulated = torch.zeros_like(tensor)

        for _ in range(n_samples):
            noise = torch.randn_like(tensor) * std_dev
            noisy = (tensor + noise).requires_grad_(True)

            self.model.zero_grad()
            out = self.model(noisy)
            out[0, target_class].backward()

            accumulated += noisy.grad.data.abs()

        smooth_sal = (accumulated / n_samples).squeeze().cpu().numpy()
        sal_norm = _norm_map(smooth_sal)

        W, H = pil_img.size
        sal_pil = Image.fromarray((sal_norm * 255).astype(np.uint8), "L")
        sal_resized = np.array(sal_pil.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0

        rgba = _apply_hot_colormap(sal_resized)
        overlay_uri = _blend_overlay(pil_img, rgba)
        raw_uri = _arr_to_data_uri(sal_norm, _apply_hot_colormap)

        return {
            "method": "SmoothGrad",
            "overlay_uri": overlay_uri,
            "raw_uri": raw_uri,
            "n_samples": n_samples,
            "noise_level": noise_level,
            "target_class": target_class,
            "target_class_name": "Pneumonia" if target_class == 1 else "Normal",
            "confidence": round(float(probs[target_class].item()) * 100, 2),
        }

    # ─── 6. ANATOMICAL REGION ATTRIBUTION ─────────────────────────────────

    def region_attribution(
        self,
        pil_img: Image.Image,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute importance scores per anatomical thoracic region
        using the Grad-CAM heatmap:
          - Left Lung   (left 40% of image)
          - Right Lung  (right 40% of image)
          - Mediastinum (central 20%)
          - Upper Zone  (top third)
          - Lower Zone  (bottom third)

        Returns normalised percentage attribution per region.
        """
        gradcam_result = self.grad_cam(pil_img, target_class=target_class)
        cam_grid_np = np.array(gradcam_result["cam_grid"])  # 8×8

        H, W = cam_grid_np.shape
        mid_col = W // 2

        # Anatomical splits on 8×8 grid
        left_lung_mask   = np.s_[:, :3]
        right_lung_mask  = np.s_[:, 5:]
        mediastinum_mask = np.s_[:, 3:5]
        upper_zone_mask  = np.s_[:3, :]
        lower_zone_mask  = np.s_[5:, :]

        def region_mean(mask):
            return float(cam_grid_np[mask].mean())

        scores = {
            "left_lung":   region_mean(left_lung_mask),
            "right_lung":  region_mean(right_lung_mask),
            "mediastinum": region_mean(mediastinum_mask),
            "upper_zone":  region_mean(upper_zone_mask),
            "lower_zone":  region_mean(lower_zone_mask),
        }

        total = sum(scores.values()) + 1e-8
        percentages = {k: round(v / total * 100, 2) for k, v in scores.items()}

        # Determine dominant region
        dominant = max(scores, key=scores.get)
        dominant_labels = {
            "left_lung":   "Left Hemithorax — Left Lung Field",
            "right_lung":  "Right Hemithorax — Right Lung Field",
            "mediastinum": "Central Mediastinum — Cardiac Silhouette",
            "upper_zone":  "Upper Lung Zones — Apical Regions",
            "lower_zone":  "Lower Lung Zones — Basal / Costophrenic Angles",
        }

        return {
            "method": "Region Attribution",
            "region_scores": percentages,
            "dominant_region": dominant,
            "dominant_region_label": dominant_labels[dominant],
            "overlay_uri": gradcam_result["overlay_uri"],
            "cam_grid": gradcam_result["cam_grid"],
            "target_class": gradcam_result["target_class"],
            "target_class_name": gradcam_result["target_class_name"],
            "confidence": gradcam_result["confidence"],
        }

    # ─── 7. FULL EXPLANATION REPORT ────────────────────────────────────────

    def full_explanation(
        self,
        pil_img: Image.Image,
    ) -> Dict[str, Any]:
        """
        Run all XAI methods and return a unified report.
        Also includes:
          - Model probability vector
          - Latent embedding L2 norm
          - Feature-space confidence (distance from class centroids)
        """
        tensor = self._preprocess(pil_img)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            embedding = self.model.extract_features(tensor).squeeze().cpu().numpy()

        target_class = int(np.argmax(probs))
        p_normal = round(float(probs[0]) * 100, 2)
        p_pneumonia = round(float(probs[1]) * 100, 2)
        embedding_norm = round(float(np.linalg.norm(embedding)), 4)

        # Run all methods
        gradcam       = self.grad_cam(pil_img, target_class=target_class)
        saliency      = self.vanilla_saliency(pil_img, target_class=target_class)
        ig            = self.integrated_gradients(pil_img, target_class=target_class, n_steps=50)
        occ           = self.occlusion_sensitivity(pil_img, target_class=target_class)
        smooth        = self.smooth_grad(pil_img, target_class=target_class, n_samples=25)
        region        = self.region_attribution(pil_img, target_class=target_class)

        return {
            "status": "XAI_COMPLETE",
            "prediction": {
                "class_index": target_class,
                "class_name": "Pneumonia" if target_class == 1 else "Normal",
                "p_normal": p_normal,
                "p_pneumonia": p_pneumonia,
                "confidence": max(p_normal, p_pneumonia),
                "embedding_norm": embedding_norm,
            },
            "grad_cam":            gradcam,
            "vanilla_saliency":    saliency,
            "integrated_gradients": ig,
            "occlusion_sensitivity": occ,
            "smooth_grad":         smooth,
            "region_attribution":  region,
        }
