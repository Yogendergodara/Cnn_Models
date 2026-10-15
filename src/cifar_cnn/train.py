import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from .config import Config
from .data import CLASSES, build_loaders
from .engine import Trainer
from .evaluate import calibrate, full_report
from .export import export_all
from .models import build_model
from .tracking import Tracker, save_history, setup_logging
from .utils import get_device, seed_everything


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/resnet_cifar10.yaml")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.resume:
        # Continue in the same folder, so best.pt from before the crash is still found
        run_dir = args.resume.parent
    else:
        run_dir = Path(cfg.output_dir) / f"{cfg.experiment}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    log = logging.getLogger("train")

    seed_everything(cfg.seed)
    device = get_device()
    meta = {"config": cfg.to_dict(), "git_commit": git_commit(),
            "torch": torch.__version__, "device": str(device)}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Run dir %s | device %s", run_dir, device)

    train_loader, val_loader, test_loader = build_loaders(cfg.data, cfg.seed)
    model = build_model(cfg.model.name, cfg.model.num_classes, cfg.model.dropout)
    log.info("Model %s: %.2fM params", cfg.model.name, sum(p.numel() for p in model.parameters()) / 1e6)

    with Tracker(cfg, run_dir) as tracker:
        trainer = Trainer(model, train_loader, val_loader, cfg, device, run_dir, tracker)
        if args.resume:
            trainer.resume(args.resume)
        history = trainer.fit()
        save_history(history, run_dir)

        # Test set is used exactly once, on the best checkpoint
        trainer.resume(run_dir / "best.pt")
        temperature = calibrate(trainer.model, val_loader, device)
        log.info("Fitted temperature %.3f on the validation set", temperature)
        report = full_report(trainer.model, test_loader, device, CLASSES, run_dir, temperature)
        tracker.log_metrics({f"test_{k}": v for k, v in report.items() if isinstance(v, float)})
        export_all(trainer.model, run_dir, device, temperature)
        tracker.log_artifacts(run_dir)


if __name__ == "__main__":      # required on Windows when num_workers > 0
    main()