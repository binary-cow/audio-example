"""
Audio classifier with pretrained EfficientNet-B0 backbone.

Architecture:
    EfficientNet-B0 (ImageNet pretrained, frozen) → AdaptiveAvgPool → Classifier Head

The first conv is adapted from 3-ch RGB to 1-ch mel spectrogram by averaging
the pretrained weights across the channel dimension. BatchNorm layers in the
frozen backbone stay in eval mode to preserve pretrained running statistics.

Total params  : ~5.1M
Trainable     : ~680K  (backbone frozen)
                ~5.1M  (backbone unfrozen)
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class AudioClassifier(nn.Module):

    def __init__(self, n_classes: int = 50, freeze_backbone: bool = True):
        super().__init__()

        # ------------------------------------------------------------------
        # Pretrained backbone
        # ------------------------------------------------------------------
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = efficientnet_b0(weights=weights)

        # Adapt first conv: 3-ch RGB → 1-ch mel spectrogram
        orig_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
        backbone.features[0][0] = new_conv

        self.backbone = backbone.features          # feature extractor
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Freeze backbone
        self._freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------
        # Classification head  (1280 → 512 → n_classes)
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_classes),
        )

    # -- helpers -----------------------------------------------------------

    @property
    def backbone_frozen(self) -> bool:
        return self._freeze_backbone

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._freeze_backbone = False

    def train(self, mode: bool = True):
        """Override to keep frozen backbone's BN in eval mode."""
        super().train(mode)
        if self._freeze_backbone and mode:
            self.backbone.eval()
        return self

    # -- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x.unsqueeze(1))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x