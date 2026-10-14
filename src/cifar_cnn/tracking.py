import csv
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def setup_logging(log_file: Path, level=logging.INFO) -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    for noisy in ("onnxscript", "onnx_ir", "torch.onnx"):   # exporter progress chatter
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Tracker:
    """MLflow if available, else TensorBoard, else files only. Same three methods either way."""

    def __init__(self, cfg, run_dir: Path):
        self.cfg, self.run_dir = cfg, run_dir
        self.backend = None

    def __enter__(self):
        try:
            import mlflow
            mlflow.set_experiment(self.cfg.experiment)
            mlflow.start_run(run_name=self.run_dir.name)
            mlflow.log_params(_flatten(self.cfg.to_dict()))
            self.backend, self._mlflow = "mlflow", mlflow
        except ImportError:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.backend, self._tb = "tensorboard", SummaryWriter(self.run_dir / "tb")
            except ImportError:
                log.warning("No mlflow or tensorboard; metrics go to train.log and history.csv only")
        return self

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if self.backend == "mlflow":
            self._mlflow.log_metrics(numeric, step=step)
        elif self.backend == "tensorboard":
            for k, v in numeric.items():
                self._tb.add_scalar(k, v, step or 0)

    def log_artifacts(self, path: Path) -> None:
        if self.backend == "mlflow":
            self._mlflow.log_artifacts(str(path))

    def __exit__(self, *exc):
        if self.backend == "mlflow":
            self._mlflow.end_run()
        elif self.backend == "tensorboard":
            self._tb.close()


def save_history(history: list[dict], run_dir: Path) -> None:
    """Per-epoch metrics as history.csv plus train-vs-validation curves as curves.png."""
    if not history:
        return
    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="Train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="Validation")
    ax_loss.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    ax_acc.plot(epochs, [h["train_acc"] for h in history], label="Train")
    ax_acc.plot(epochs, [h["val_acc"] for h in history], label="Validation")
    ax_acc.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=120)
    plt.close(fig)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out
