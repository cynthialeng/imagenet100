import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import BATCH_SIZE, HW2Dataset
from model import IMAGE_SIZE, NUM_CLASSES, LinearClassifier, TwoHiddenMLP


OUTPUTS = Path(__file__).resolve().parent / "outputs"


def evaluate_checkpoint(path, model_class, expected_type, dataset):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != dataset.classes:
        raise ValueError(f"Class order mismatch in {path}")
    if checkpoint["image_size"] != IMAGE_SIZE:
        raise ValueError(f"Image size mismatch in {path}")
    if checkpoint["model_type"] != expected_type:
        raise ValueError(f"Model type mismatch in {path}")
    model = model_class()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

    # No optimizer or backward pass: the test images never update the model.
    with torch.no_grad():
        for images, labels in loader:
            scores = model(images)
            loss_sum += loss_fn(scores, labels).item()
            predictions = scores.argmax(dim=1)
            for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                confusion[actual, predicted] += 1

    correct = int(confusion.diag().sum())
    count = len(dataset)
    return {
        "checkpoint": str(path.resolve()),
        "selected_epoch": checkpoint["epoch"],
        "test_images": count,
        "correct": correct,
        "incorrect": count - correct,
        "loss": loss_sum / count,
        "accuracy": correct / count,
        "error_rate": (count - correct) / count,
        "confusion_matrix": confusion.tolist(),
    }


def main():
    dataset = HW2Dataset("test")
    report = {"classes": dataset.classes, "models": {}}
    experiments = [
        ("linear_baseline", LinearClassifier, "linear"),
        ("two_hidden", TwoHiddenMLP, "mlp2"),
    ]
    # Check both files before starting the comparison.
    for folder, _, _ in experiments:
        path = OUTPUTS / folder / "best_model.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    print("Class order:", dict(enumerate(dataset.classes)))
    for folder, model_class, model_type in experiments:
        result = evaluate_checkpoint(OUTPUTS / folder / "best_model.pt",
                                     model_class, model_type, dataset)
        report["models"][folder] = result
        print(f"\n{folder} (selected epoch {result['selected_epoch']})")
        print(f"Test loss: {result['loss']:.4f}")
        print(f"Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['test_images']})")
        print(f"Error rate: {result['error_rate']:.1%}")
        print("Confusion matrix: rows = actual, columns = predicted")
        for row in result["confusion_matrix"]:
            print(" ", row)

    linear = report["models"]["linear_baseline"]
    mlp = report["models"]["two_hidden"]
    difference = 100 * (mlp["accuracy"] - linear["accuracy"])
    report["mlp_accuracy_difference_percentage_points"] = difference
    print(f"\nMLP minus benchmark accuracy: {difference:+.1f} percentage points")
    path = OUTPUTS / "test_results.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
