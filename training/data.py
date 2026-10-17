import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from .model import IMAGE_SIZE, NUM_CLASSES
except ImportError:
    from model import IMAGE_SIZE, NUM_CLASSES


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hw2"
BATCH_SIZE = 32


def prepare_image(image):
    """Convert a PIL image to a float tensor shaped (3, 32, 32)."""
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    # Pixel values start at 0–255; scale them to 0–1.
    pixels = np.array(image, dtype=np.float32) / 255.0
    # PIL uses (height, width, channels); PyTorch expects channels first.
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


class HW2Dataset(Dataset):
    def __init__(self, split, data_dir=DATA_DIR):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val', or 'test'")
        self.data_dir = Path(data_dir)
        manifest = json.loads((self.data_dir / "metadata.json").read_text())
        self.classes = manifest["classes"]
        if len(self.classes) != NUM_CLASSES:
            raise ValueError("Dataset class count does not match the model")
        # Use the manifest labels so the order stays tabby=0,
        # cauliflower=1, computer keyboard=2, regardless of folder sorting.
        self.samples = [row for row in manifest["images"] if row["split"] == split]
        if not self.samples:
            raise ValueError(f"No images found for split {split!r}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(self.data_dir / sample["path"]) as image:
            tensor = prepare_image(image)
        return tensor, sample["class_index"]


def make_loaders(batch_size=BATCH_SIZE):
    train_data = HW2Dataset("train")
    val_data = HW2Dataset("val")
    # Shuffle training examples each epoch to mix the three classes.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader