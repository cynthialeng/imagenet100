"""HW3 test evaluation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.data import DATA_DIR, HW2Dataset
from training.model import IMAGE_SIZE
from .models import LocalGlobalClassifier
from .train import gate_statistics


ROOT = Path(__file__).resolve().parent / "outputs"


def evaluate_checkpoint(path, dataset, manifest_hash):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["manifest_sha256"] != manifest_hash:
        raise ValueError(f"Dataset manifest changed since training: {path}")
    if checkpoint["classes"] != dataset.classes or checkpoint["image_size"] != IMAGE_SIZE:
        raise ValueError(f"Class order or preprocessing size mismatch: {path}")
    summary = json.loads((path.parent / "summary.json").read_text())
    if summary["status"] != "completed" or summary["best"]["epoch"] != checkpoint["epoch"]:
        raise ValueError(f"Checkpoint does not match the validation-selected epoch: {path}")
    pooling = config.get("global_pooling", "avg")
    model = LocalGlobalClassifier(checkpoint["variant"], pooling=pooling)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    confusion = torch.zeros(len(dataset.classes), len(dataset.classes), dtype=torch.long)
    loss_sum, predictions, gates = 0.0, [], []
    offset = 0
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=config["batch_size"], shuffle=False):
            scores, alpha = model.forward_with_gate(images)
            loss = nn.functional.cross_entropy(scores, labels, reduction="sum")
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite test loss: {path}")
            loss_sum += loss.item()
            predicted = scores.argmax(dim=1)
            if alpha is not None:
                gates.append(alpha.flatten())
            for index, (actual, guess) in enumerate(zip(labels.tolist(), predicted.tolist())):
                confusion[actual, guess] += 1
                predictions.append({"path": dataset.samples[offset + index]["path"],
                                    "actual_class": dataset.classes[actual],
                                    "predicted_class": dataset.classes[guess],
                                    "cnn_gate": float(alpha[index, 0]) if alpha is not None else None})
            offset += len(labels)
    count = len(dataset)
    correct = int(confusion.diag().sum())
    result = {
        "variant": checkpoint["variant"], "pooling": pooling, "seed": config["seed"],
        "selected_epoch": checkpoint["epoch"], "test_images": count,
        "loss": loss_sum / count, "accuracy": correct / count,
        "error_rate": (count - correct) / count, "correct": correct,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_order": dataset.classes,
        "confusion_matrix_axes": "rows = actual, columns = predicted",
        "gate": gate_statistics(torch.cat(gates)) if gates else None,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "manifest_sha256": manifest_hash,
        "predictions": predictions,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="*", type=Path,
                        default=[ROOT / name for name in ("first_comparison", "cnn_max_pool", "fusion_max_pool")])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis")
    args = parser.parse_args()
    paths = []
    for folder in args.folders:
        found = sorted(folder.glob("seed_*/*/best_model.pt"))
        if not found:
            parser.error(f"No checkpoints found in {folder}")
        paths.extend(found)
    if len(set(path.resolve() for path in paths)) != len(paths):
        parser.error("Duplicate checkpoint paths")
    dataset = HW2Dataset("test", args.data_dir)
    manifest_hash = hashlib.sha256((args.data_dir / "metadata.json").read_bytes()).hexdigest()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results = []
    for path in paths:
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        result = evaluate_checkpoint(path, dataset, manifest_hash)
        if hashlib.sha256(path.read_bytes()).hexdigest() != before:
            raise RuntimeError(f"Checkpoint changed during evaluation: {path}")
        results.append(result)
        print(f"{result['variant']:16} {result['pooling']:3} seed={result['seed']} | "
              f"epoch {result['selected_epoch']:3} | test loss {result['loss']:.4f} | "
              f"accuracy {result['accuracy']:.1%} ({result['correct']}/{result['test_images']}) | "
              f"error {result['error_rate']:.1%}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"classes": dataset.classes, "selection": "lowest validation loss, before test evaluation",
              "note": "Same test split used in HW2; not a new independent holdout.", "results": results}
    (args.output_dir / "test_results.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["variant", "pooling", "seed", "selected_epoch", "test_images", "loss", "accuracy", "error_rate", "correct"]
    with (args.output_dir / "test_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved test_results.json and test_summary.csv to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
