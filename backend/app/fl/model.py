"""
app/fl/model.py

High-performance Convolutional Neural Network for Pneumonia detection in chest X-rays.
Features double-convolution blocks, Batch Normalization, LeakyReLU activations,
and Adaptive Average Pooling to dynamically support arbitrary input resolutions
(28x28, 64x64, 224x224, or user-uploaded web images).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Dual Convolution block with Batch Normalization and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PneumoniaCNN(nn.Module):
    """
    Robust medical CNN for PneumoniaMNIST classification.
    Produces high feature separability to enable both high classification accuracy (>93%)
    and latent embedding extraction for dataset poisoning detection.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Feature Extractor: 3 stages (1 -> 32 -> 64 -> 128)
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        # Adaptive pool guarantees fixed output (128 * 2 * 2 = 512) regardless of input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        # Regularized Dense Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation vector (512-dim) for poison detection."""
        feats = self.features(x)
        pooled = self.adaptive_pool(feats)
        return torch.flatten(pooled, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        pooled = self.adaptive_pool(feats)
        return self.classifier(pooled)