"""Calibrate an already-trained run without retraining.

Usage:  python -m cifar_cnn.calibrate runs/<run_dir>
Fits a temperature on the validation set using best.pt, then rewrites test_report.json
(plus its plots) and the temperature in model_meta.json. The exported weights are unchanged.
"""
import argparse
import json
import logging
from pathlib import Path

import torch

from .config import Config
from .data import CLASSES, build_loaders
from .evaluate import calibrate, full_report
from .models import build_model
from .utils import get_device

log = logging.getLogger("calibrate")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    run_dir = ap.parse_args().run

    cfg = Config.from_dict(json.loads((run_dir / "run_meta.json").read_text())["config"])
    device = get_device()
    _, val_loader, test_loader = build_loaders(cfg.data, cfg.seed)
    model = build_model(cfg.model.name, cfg.model.num_classes, cfg.model.dropout).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])

    temperature = calibrate(model, val_loader, device)
    report = full_report(model, test_loader, device, CLASSES, run_dir, temperature)

    meta_path = run_dir / "model_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["temperature"] = temperature
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("temperature %.3f | ECE %.4f -> %.4f | accuracy %.4f", temperature,
             report["ece_uncalibrated"], report["ece"], report["accuracy"])


if __name__ == "__main__":
    main()
