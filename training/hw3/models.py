"""HW3 local–global ablations."""

import torch
from torch import nn

from training.model import HIDDEN_SIZE, IMAGE_SIZE, NUM_CLASSES


FEATURE_SIZE = 64
VARIANTS = ("mlp_only", "cnn_only", "fixed_fusion", "learned_fusion")


class GlobalFeatures(nn.Sequential):
    """The HW2 MLP's hidden layers, before its final classifier."""

    def __init__(self):
        super().__init__(
            nn.Flatten(start_dim=1),
            nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, FEATURE_SIZE),
            nn.ReLU(),
        )


class LocalFeatures(nn.Sequential):
    """Spatial filters followed by a 64-dimensional whole-image summary."""

    def __init__(self, pooling="avg"):
        if pooling not in ("avg", "max"):
            raise ValueError("pooling must be 'avg' or 'max'")
        global_pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        super().__init__(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # (batch, 16, 16, 16)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # (batch, 32, 8, 8)
            global_pool,                     # (batch, 32, 1, 1)
            nn.Flatten(start_dim=1),
            nn.Linear(32, FEATURE_SIZE),
            nn.ReLU(),
        )


class LocalGlobalClassifier(nn.Module):
    """Four controlled variants sharing feature and classifier definitions.

    forward() returns raw class scores for cross-entropy loss.
    forward_with_gate() additionally returns alpha shaped (batch, 1), or None
    for a single-branch model. Alpha is the CNN weight, not a class probability.
    """

    def __init__(self, variant="learned_fusion", pooling="avg"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}. Choose from {VARIANTS}")
        self.variant = variant
        self.pooling = pooling
        self.global_branch = GlobalFeatures() if variant != "cnn_only" else None
        self.local_branch = LocalFeatures(pooling) if variant != "mlp_only" else None
        self.classifier = nn.Linear(FEATURE_SIZE, NUM_CLASSES)

        # Create the gate last: fixed and learned fusion have identical initial
        # branch/head weights when constructed with the same random seed.
        self.gate = None
        if variant == "learned_fusion":
            self.gate = nn.Linear(2 * FEATURE_SIZE, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward_with_gate(self, images):
        if self.variant == "mlp_only":
            return self.classifier(self.global_branch(images)), None
        if self.variant == "cnn_only":
            return self.classifier(self.local_branch(images)), None

        local = self.local_branch(images)
        global_features = self.global_branch(images)
        if self.gate is None:
            alpha = local.new_full((images.shape[0], 1), 0.5)
        else:
            combined = torch.cat((local, global_features), dim=1)
            alpha = torch.sigmoid(self.gate(combined))

        # (batch, 1) broadcasts across all 64 features for each image.
        fused = alpha * local + (1 - alpha) * global_features
        return self.classifier(fused), alpha

    def forward(self, images):
        logits, _ = self.forward_with_gate(images)
        return logits


if __name__ == "__main__":
    images = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    print("Input:", tuple(images.shape))
    for variant in VARIANTS:
        torch.manual_seed(42)
        model = LocalGlobalClassifier(variant)
        with torch.no_grad():
            scores, alpha = model.forward_with_gate(images)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{variant}: output={tuple(scores.shape)}, parameters={parameters:,}")
        if alpha is not None:
            print("  CNN gate weights:", alpha.flatten().tolist())
