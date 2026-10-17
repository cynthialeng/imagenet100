"""Plot Training Curves"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="Path to the training history.csv")
    args = parser.parse_args()
    with args.history.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        parser.error("The history file contains no epochs")

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_accuracy = [100 * float(row["train_accuracy"]) for row in rows]
    val_accuracy = [100 * float(row["val_accuracy"]) for row in rows]
    best_index = min(range(len(rows)), key=lambda index: val_loss[index])
    best_epoch = epochs[best_index]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(epochs, train_loss, label="Train", color="tab:blue")
    axes[0].plot(epochs, val_loss, label="Val", color="tab:orange")
    axes[0].scatter([best_epoch], [val_loss[best_index]], color="black", zorder=3,
                    label=f"Lowest val loss (epoch {best_epoch})")
    axes[0].set(title="Loss vs Epoch", xlabel="Epoch", ylabel="Cross-entropy loss")

    axes[1].plot(epochs, train_accuracy, label="Train", color="tab:blue")
    axes[1].plot(epochs, val_accuracy, label="Val", color="tab:orange")
    axes[1].set(title="Accuracy vs Epoch", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("ImageNet100 3-class MLP", fontsize=14)
    output = args.history.parent / "training_curves.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)

    print(f"Plotted {len(rows)} epochs")
    print(f"Lowest validation loss: {val_loss[best_index]:.4f} at epoch {best_epoch}")
    print(f"Validation accuracy at that epoch: {val_accuracy[best_index]:.1f}%")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
