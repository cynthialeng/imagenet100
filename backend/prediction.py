"""Inference using the HW2 checkpoint"""

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from training.data import prepare_image
from training.model import IMAGE_SIZE, NUM_CLASSES, TwoHiddenMLP


CHECKPOINT_PATH = (Path(__file__).resolve().parent.parent / "training" / "outputs"
                   / "two_hidden" / "best_model.pt")


@lru_cache(maxsize=1)
def load_model(path=CHECKPOINT_PATH):
    # Load once on the first prediction, then reuse the model for later requests.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["model_type"] != "mlp2" or checkpoint["image_size"] != IMAGE_SIZE:
        raise ValueError("Checkpoint architecture does not match TwoHiddenMLP")
    classes = checkpoint["classes"]
    if len(classes) != NUM_CLASSES or len(set(classes)) != NUM_CLASSES:
        raise ValueError("Invalid checkpoint classes")
    model = TwoHiddenMLP()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


def predict_image(content, checkpoint_path=CHECKPOINT_PATH):
    try:
        with Image.open(BytesIO(content)) as image:
            image_tensor = prepare_image(image).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("Provide a valid, readable image file") from exc

    model, classes = load_model(checkpoint_path)
    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = logits.softmax(dim=1)[0]
    class_index = int(probabilities.argmax())
    return {
        "predicted_class": classes[class_index],
        "class_index": class_index,
        "probabilities": [
            {"class_name": name, "probability": float(probabilities[index])}
            for index, name in enumerate(classes)
        ],
        "supported_classes": classes,
    }
