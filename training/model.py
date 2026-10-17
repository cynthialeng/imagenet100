from torch import nn


IMAGE_SIZE = 32
NUM_CLASSES = 3
HIDDEN_SIZE = 128


class LinearClassifier(nn.Module):
    """Shallow benchmark"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, NUM_CLASSES),
        )

    def forward(self, images):
        return self.layers(images)


class ImageMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Each RGB image has 3 * 32 * 32 = 3,072 input values.
            nn.Flatten(start_dim=1),
            nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            # Return one raw score (logit) per class.
            nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
        )

    def forward(self, images):
        return self.layers(images)


class TwoHiddenMLP(nn.Module):
    """Two-hidden-layer experiment"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, images):
        return self.layers(images)
