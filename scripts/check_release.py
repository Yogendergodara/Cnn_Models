"""Release gate: exit 1 unless a trained run is good enough to ship.

Usage:  python scripts/check_release.py runs/<run_dir> [--baseline runs/<current_prod_run>]
Checks, against the run's test_report.json:
  - accuracy at least MIN_ACCURACY (and at least the baseline's, if given)
  - no class loses more than MAX_CLASS_DROP versus the baseline
  - calibration error under MAX_ECE
  - the exported files exist
"""
import argparse
import json
import sys
from pathlib import Path

MIN_ACCURACY = 0.85
MAX_CLASS_DROP = 0.02
MAX_ECE = 0.05
REQUIRED_FILES = ("model.onnx", "model_meta.json", "best.pt", "test_report.json")


def check(run: Path, baseline: Path | None) -> list[str]:
    if not run.is_dir():
        return [f"run folder not found: {run} (pass a runs/<name> folder that contains the exported files)"]
    problems = []
    for name in REQUIRED_FILES:
        if not (run / name).exists():
            problems.append(f"missing {name}")
    if len(problems) == len(REQUIRED_FILES):
        return [f"{run} contains none of the release files: wrong folder or training/export never finished"]
    if problems:
        return problems

    new = json.loads((run / "test_report.json").read_text())
    if new["accuracy"] < MIN_ACCURACY:
        problems.append(f"accuracy {new['accuracy']:.4f} is below the minimum {MIN_ACCURACY}")
    if new["ece"] > MAX_ECE:
        problems.append(f"ECE {new['ece']:.4f} is above the maximum {MAX_ECE}")

    if baseline is not None:
        if not (baseline / "test_report.json").is_file():
            return problems + [f"baseline has no test_report.json: {baseline}"]
        old = json.loads((baseline / "test_report.json").read_text())
        if new["accuracy"] < old["accuracy"]:
            problems.append(f"accuracy {new['accuracy']:.4f} is below the baseline {old['accuracy']:.4f}")
        for cls, old_acc in old["per_class_accuracy"].items():
            if cls not in new["per_class_accuracy"]:
                problems.append(f"class {cls!r} is missing from the new report")
                continue
            drop = old_acc - new["per_class_accuracy"][cls]
            if drop > MAX_CLASS_DROP:
                problems.append(f"class {cls!r} dropped {drop:.3f} versus the baseline")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--baseline", type=Path, default=None)
    args = ap.parse_args()

    problems = check(args.run, args.baseline)
    if problems:
        print("RELEASE BLOCKED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"Release gate passed for {args.run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
