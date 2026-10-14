import numpy as np


def psi(reference: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    """Population stability index. <0.1 stable, 0.1-0.2 watch, >0.2 drifted."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference) + 1e-6
    live_pct = np.histogram(live, edges)[0] / len(live) + 1e-6
    return float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))


def energy_score(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Higher = more likely out-of-distribution. Better than max softmax for OOD detection."""
    m = logits.max(1, keepdims=True)
    return -(T * np.log(np.exp((logits - m) / T).sum(1)) + m.squeeze(1))