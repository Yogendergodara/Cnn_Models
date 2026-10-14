import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_release import check  # noqa: E402

from cifar_cnn.monitoring.drift import energy_score, psi  # noqa: E402


def _make_run(path: Path, accuracy: float, ece: float = 0.02, per_class: float | None = None):
    path.mkdir()
    for name in ("model.onnx", "model_meta.json", "best.pt"):
        (path / name).write_text("x")
    (path / "test_report.json").write_text(json.dumps({
        "accuracy": accuracy, "ece": ece,
        "per_class_accuracy": {"cat": per_class if per_class is not None else accuracy},
    }))
    return path


def test_release_gate_passes_a_good_run(tmp_path):
    assert check(_make_run(tmp_path / "good", 0.93), None) == []


def test_release_gate_blocks_low_accuracy_and_missing_files(tmp_path):
    assert check(_make_run(tmp_path / "bad", 0.70), None)
    (tmp_path / "empty").mkdir()
    assert "missing model.onnx" in check(tmp_path / "empty", None)


def test_release_gate_blocks_regression_against_baseline(tmp_path):
    old = _make_run(tmp_path / "old", 0.93)
    new = _make_run(tmp_path / "new", 0.90)
    assert any("baseline" in p for p in check(new, old))


def test_psi_is_zero_for_same_data_and_large_for_shifted_data():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.1
    assert psi(ref, rng.normal(2, 1, 5000)) > 0.2


def test_energy_score_is_higher_for_flat_logits():
    confident = np.array([[10.0, 0, 0, 0]])
    flat = np.array([[0.1, 0.1, 0.1, 0.1]])
    assert energy_score(flat)[0] > energy_score(confident)[0]
