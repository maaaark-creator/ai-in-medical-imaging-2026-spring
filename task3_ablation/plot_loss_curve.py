import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)\s+summary")
TRAIN_RE = re.compile(r"train total=([0-9.]+)")
VAL_RE = re.compile(r"val\s+total=([0-9.]+)")


def parse_log(log_path: Path):
    epochs = []
    train_losses = []
    val_losses = []

    current_epoch = None
    pending_train = None

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            pending_train = None
            continue

        if current_epoch is None:
            continue

        train_match = TRAIN_RE.search(line)
        if train_match:
            pending_train = float(train_match.group(1))
            continue

        val_match = VAL_RE.search(line)
        if val_match and pending_train is not None:
            epochs.append(current_epoch)
            train_losses.append(pending_train)
            val_losses.append(float(val_match.group(1)))
            current_epoch = None
            pending_train = None

    return epochs, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description="Plot train/val loss curve from training log.")
    parser.add_argument("log_path", type=Path, help="Path to the training log file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("loss_curve.png"),
        help="Output image path",
    )
    args = parser.parse_args()

    epochs, train_losses, val_losses = parse_log(args.log_path)
    if not epochs:
        raise SystemExit(f"No epoch summaries found in {args.log_path}")

    best_idx = min(range(len(val_losses)), key=val_losses.__getitem__)
    best_epoch = epochs[best_idx]
    best_val = val_losses[best_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", linewidth=2, label="Train Loss")
    plt.plot(epochs, val_losses, marker="s", linewidth=2, label="Val Loss")
    plt.scatter([best_epoch], [best_val], color="crimson", zorder=5, label=f"Best Val (Epoch {best_epoch})")
    plt.annotate(
        f"{best_val:.6f}",
        xy=(best_epoch, best_val),
        xytext=(8, -14),
        textcoords="offset points",
        color="crimson",
    )

    plt.title(f"Loss Curve: {args.log_path.name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    print(f"Saved loss curve to: {args.output}")
    print(f"Best val loss: {best_val:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
