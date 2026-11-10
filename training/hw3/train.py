"""
Controlled HW3 ablations. This runner uses only training and validation data, never the test set.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.data import DATA_DIR, HW2Dataset
from training.model import IMAGE_SIZE
from training.train import train_epoch
from .models import LocalFeatures, LocalGlobalClassifier, VARIANTS


def build_model(variant, seed, pooling="avg"):
    """Give shared components identical initial weights across all ablations."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        reference = LocalGlobalClassifier("mlp_only")
        torch.manual_seed(seed + 1)
        local = LocalFeatures(pooling)
        model = LocalGlobalClassifier(variant, pooling=pooling)
        if model.global_branch is not None:
            model.global_branch.load_state_dict(reference.global_branch.state_dict())
        if model.local_branch is not None:
            model.local_branch.load_state_dict(local.state_dict())
        model.classifier.load_state_dict(reference.classifier.state_dict())
    return model


def make_loaders(train_data, val_data, seed, batch_size):
    # A private RNG keeps batch order independent of model initialization.
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def gate_statistics(values):
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "max": values.max().item(),
        "fraction_below_0_05": (values < 0.05).float().mean().item(),
        "fraction_above_0_95": (values > 0.95).float().mean().item(),
    }


def validate(model, loader, classes):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    gates, labels_for_gates = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits, alpha = model.forward_with_gate(images)
            loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
            if not torch.isfinite(loss):
                raise ValueError("Validation loss is not finite")
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += len(labels)
            if alpha is not None:
                gates.append(alpha.flatten())
                labels_for_gates.append(labels)
    diagnostics = None
    if gates:
        values = torch.cat(gates)
        labels = torch.cat(labels_for_gates)
        diagnostics = gate_statistics(values)
        diagnostics["by_class"] = {
            name: gate_statistics(values[labels == index])
            for index, name in enumerate(classes) if (labels == index).any()
        }
    return total_loss / count, correct / count, diagnostics


def run_experiment(variant, seed, args, train_data, val_data, manifest_hash):
    output = args.output_dir / f"seed_{seed}" / variant
    output.mkdir(parents=True)
    config = {
        "variant": variant, "seed": seed, "epochs": args.epochs,
        "global_pooling": args.pooling,
        "learning_rate": args.lr, "batch_size": args.batch_size,
        "optimizer": "SGD", "momentum": 0, "weight_decay": 0,
        "device": "cpu", "threads": args.threads,
        "deterministic_algorithms": True,
        "data_dir": str(args.data_dir.resolve()), "manifest_sha256": manifest_hash,
        "classes": train_data.classes, "image_size": IMAGE_SIZE,
        "train_images": len(train_data), "val_images": len(val_data),
        "preprocessing": "RGB, bilinear resize 32x32, float32 pixels / 255",
        "checkpoint_selection": "lowest validation cross-entropy loss",
        "torch_version": str(torch.__version__), "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    start = perf_counter()
    try:
        torch.manual_seed(seed)
        model = build_model(variant, seed, pooling=args.pooling)
        train_loader, val_loader = make_loaders(train_data, val_data, seed, args.batch_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        best = None
        epoch_times = []
        fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "seconds"]
        with (output / "history.csv").open("w", newline="") as history_file, \
                (output / "gate_history.jsonl").open("w") as gate_file:
            writer = csv.DictWriter(history_file, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                epoch_start = perf_counter()
                train_loss, train_accuracy = train_epoch(model, train_loader, loss_fn, optimizer)
                val_loss, val_accuracy, gate = validate(model, val_loader, train_data.classes)
                seconds = perf_counter() - epoch_start
                epoch_times.append(seconds)
                row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                           val_loss=val_loss, val_accuracy=val_accuracy, seconds=seconds)
                writer.writerow(row)
                history_file.flush()
                if gate is not None:
                    gate_file.write(json.dumps({"epoch": epoch, **gate}) + "\n")
                    gate_file.flush()
                if best is None or val_loss < best["val_loss"]:
                    best = {**row, "gate": gate}
                    torch.save({"model_state_dict": model.state_dict(), "config": config,
                                "variant": variant, "classes": train_data.classes,
                                "image_size": IMAGE_SIZE, "epoch": epoch,
                                "val_loss": val_loss}, output / "best_model.pt")
                gate_text = f" | CNN gate {gate['mean']:.3f} ± {gate['std']:.3f}" if gate else ""
                print(f"[{variant}, seed {seed}] {epoch:03d}/{args.epochs} | "
                      f"train {train_loss:.4f}, {train_accuracy:.1%} | "
                      f"val {val_loss:.4f}, {val_accuracy:.1%} | {seconds:.2f}s{gate_text}", flush=True)
        summary = {"variant": variant, "seed": seed, "global_pooling": args.pooling, "status": "completed",
                   "parameters": parameters, "best": best,
                   "total_seconds": perf_counter() - start,
                   "mean_epoch_seconds": sum(epoch_times) / len(epoch_times)}
    except Exception as exc:
        # Keep incomplete histories and record failures instead of silently dropping them.
        summary = {"variant": variant, "seed": seed, "status": "failed",
                   "error": f"{type(exc).__name__}: {exc}", "total_seconds": perf_counter() - start}
        print(f"[{variant}, seed {seed}] FAILED: {summary['error']}", flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--pooling", choices=["avg", "max"], default="avg",
                        help="Global pooling in the CNN branch (default: avg)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "first_comparison")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch size, threads, and learning rate must be positive and finite")
    if len(set(args.variants)) != len(args.variants) or len(set(args.seeds)) != len(args.seeds):
        parser.error("variants and seeds must not contain duplicates")
    if any(seed < 0 or seed > 2**32 - 2 for seed in args.seeds):
        parser.error("seeds must be between 0 and 4294967294")
    if args.output_dir.exists():
        parser.error("Output folder already exists; choose a new --output-dir")
    train_data = HW2Dataset("train", args.data_dir)
    val_data = HW2Dataset("val", args.data_dir)
    manifest_hash = hashlib.sha256((args.data_dir / "metadata.json").read_bytes()).hexdigest()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True)
    summaries = []
    for seed in args.seeds:
        for variant in args.variants:
            summaries.append(run_experiment(variant, seed, args, train_data, val_data, manifest_hash))
            (args.output_dir / "comparison.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print("\nValidation comparison (checkpoint selected by loss):")
    for summary in summaries:
        label = f"{summary['variant']:16} seed={summary['seed']}"
        if summary["status"] == "completed":
            best = summary["best"]
            print(f"{label} | epoch {best['epoch']:3d} | loss {best['val_loss']:.4f} | "
                  f"accuracy {best['val_accuracy']:.1%} | parameters {summary['parameters']:,}")
        else:
            print(f"{label} | FAILED: {summary['error']}")
    print(f"Saved to {args.output_dir.resolve()}. No test images were evaluated.")
    if any(row["status"] == "failed" for row in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
