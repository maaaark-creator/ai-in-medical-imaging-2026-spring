import argparse
import csv
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_task3_line"
DEFAULT_LOG_PATH = DEFAULT_OUTPUT_DIR / "training_log.csv"


def read_training_log(log_path: Path) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    with log_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"epoch", "train_loss", "val_loss"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required column(s) in {log_path}: {missing}")

        for row in reader:
            history.append(
                {
                    "epoch": float(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                }
            )

    if not history:
        raise ValueError(f"No rows found in {log_path}")
    return history


def plot_loss_curve(history: list[dict[str, float]], output_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping loss curve.")
        return None

    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_losses, label="Train Loss", marker="o", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Validation Loss", marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Task 3 Training Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "loss_curve.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train and validation loss curves from training_log.csv."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Path to training_log.csv. Defaults to {DEFAULT_LOG_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save loss_curve.png. Defaults to {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = read_training_log(args.log_path)
    output_path = plot_loss_curve(history, args.output_dir)
    if output_path is not None:
        print(f"Saved loss curve to {output_path}")


if __name__ == "__main__":
    main()
