import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")             # no window: works on servers and in CI
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


@torch.no_grad()
def collect_logits(model, loader, device, tta: bool = True):
    """Raw logits as (views, N, classes): the plain image and, with tta, its horizontal flip."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        views = [model(images)]
        if tta:
            views.append(model(torch.flip(images, dims=[3])))
        all_logits.append(torch.stack(views).float().cpu())
        all_labels.append(labels)
    return torch.cat(all_logits, dim=1), torch.cat(all_labels)


def probs_from_logits(logits, temperature: float = 1.0):
    return F.softmax(logits / temperature, dim=2).mean(0)


def predict(model, loader, device, tta: bool = True, temperature: float = 1.0):
    logits, labels = collect_logits(model, loader, device, tta)
    return probs_from_logits(logits, temperature).numpy(), labels.numpy()


def fit_temperature(logits, labels) -> float:
    """Single scalar T minimising NLL of the (TTA-averaged) probabilities. Never changes the argmax."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.nll_loss(torch.log(probs_from_logits(logits, log_t.exp()) + 1e-12), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().detach())


def calibrate(model, val_loader, device) -> float:
    """Fit the temperature on the validation set (the test set stays untouched)."""
    logits, labels = collect_logits(model, val_loader, device, tta=True)
    return fit_temperature(logits, labels)


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def full_report(model, loader, device, classes, out_dir: Path, temperature: float = 1.0) -> dict:
    logits, labels = collect_logits(model, loader, device, tta=True)
    labels = labels.numpy()
    probs = probs_from_logits(logits, temperature).numpy()
    preds = probs.argmax(1)

    report = classification_report(labels, preds, target_names=classes, output_dict=True, digits=4,
                                   zero_division=0)
    cm = confusion_matrix(labels, preds)
    per_class_acc = {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(classes)}

    result = {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "ece": expected_calibration_error(probs, labels),
        "ece_uncalibrated": expected_calibration_error(probs_from_logits(logits).numpy(), labels),
        "temperature": temperature,
        "per_class_accuracy": per_class_acc,
        "classification_report": report,
    }
    (out_dir / "test_report.json").write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Test confusion matrix (acc {result['accuracy']:.2%})")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Save the 25 most confident mistakes: the fastest way to find label noise and real weaknesses
    wrong = np.where(preds != labels)[0]
    worst = wrong[np.argsort(-probs[wrong].max(1))][:25]
    np.save(out_dir / "worst_mistakes_idx.npy", worst)
    return result


@torch.no_grad()
def benchmark_latency(model, device, batch_size: int = 1, runs: int = 200) -> dict:
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    for _ in range(20):                    # warm-up
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
            "imgs_per_sec": float(batch_size * 1000 / times.mean())}