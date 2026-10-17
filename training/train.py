import argparse
import csv
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

try:
    from .data import make_loaders
    from .model import IMAGE_SIZE, ImageMLP, LinearClassifier, TwoHiddenMLP
except ImportError:
    from data import make_loaders
    from model import IMAGE_SIZE, ImageMLP, LinearClassifier, TwoHiddenMLP


def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        optimizer.zero_grad()              # Clear gradients from the previous batch
        scores = model(images)             # Forward propagation
        loss = loss_fn(scores, labels)     
        if not torch.isfinite(loss):
            raise ValueError("Loss is not finite. Try a smaller learning rate.")
        loss.backward()                    # Backpropagation
        optimizer.step()                   # Update weights and biases using SGD

        total_loss += loss.item() * len(labels)
        correct += (scores.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    
    return total_loss / total, correct / total


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            scores = model(images)
            loss = loss_fn(scores, labels)
            total_loss += loss.item() * len(labels)
            correct += (scores.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mlp", "mlp2", "linear"], default="mlp",
                        help="One-hidden-layer MLP, two-hidden-layer MLP, or linear benchmark")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="L2 regularization strength (0 reproduces the baseline)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "first_run")
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        parser.error("epochs and learning rate must be positive")
    if args.weight_decay < 0:
        parser.error("weight decay must be nonnegative")
    if args.output_dir.exists():
        parser.error("Output folder already exists; choose a new --output-dir")

    torch.manual_seed(42)
    train_loader, val_loader = make_loaders()
    model_classes = {"mlp": ImageMLP, "mlp2": TwoHiddenMLP, "linear": LinearClassifier}
    model = model_classes[args.model]()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    print(f"Model: {args.model} | Learning rate: {args.lr} | Weight decay: {args.weight_decay}")
    args.output_dir.mkdir(parents=True)
    best_val_loss = float("inf")

    with (args.output_dir / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "seconds"
        ])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = perf_counter()
            train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn)
            seconds = perf_counter() - start
            writer.writerow(dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc,
                                 val_loss=val_loss, val_accuracy=val_acc, seconds=seconds))
            file.flush()
            # Save the epoch that performs best by validation loss.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model,
                    "classes": train_loader.dataset.classes,
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "seed": 42,
                }, args.output_dir / "best_model.pt")
            print(f"Epoch {epoch:02d}/{args.epochs} | "
                  f"train loss {train_loss:.4f}, acc {train_acc:.1%} | "
                  f"val loss {val_loss:.4f}, acc {val_acc:.1%} | {seconds:.2f}s", flush=True)
    print(f"Saved history.csv and best_model.pt to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
