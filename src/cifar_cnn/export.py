import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from .data import CIFAR10_MEAN, CIFAR10_STD, CLASSES

log = logging.getLogger(__name__)


def export_all(model, run_dir: Path, device) -> None:
    model = getattr(model, "_orig_mod", model).float().cpu().eval()
    # Batch of 2, not 1: torch.export specialises a size-1 dim to a constant
    example = torch.randn(2, 3, 32, 32)
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        ref = model(x).numpy()

    # ONNX (what the API serves), with a dynamic batch dimension.
    # verbose=False: the exporter's progress lines contain emoji that crash on Windows cp1252 consoles.
    # external_data=False: keep weights inside model.onnx instead of a separate model.onnx.data file.
    onnx_path = run_dir / "model.onnx"
    batch = torch.export.Dim("batch", min=1, max=1024)
    torch.onnx.export(model, (example,), onnx_path, input_names=["images"], output_names=["logits"],
                      dynamic_shapes={"x": {0: batch}}, opset_version=18,
                      dynamo=True, external_data=False, verbose=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"images": x.numpy()})[0]
    np.testing.assert_allclose(ref, got, rtol=1e-3, atol=1e-4)

    # Python-side format: torch.export (.pt2). TorchScript is not supported on Python 3.14+.
    exported = torch.export.export(model, (example,), dynamic_shapes={"x": {0: batch}})
    torch.export.save(exported, run_dir / "model.pt2")
    np.testing.assert_allclose(ref, exported.module()(x).detach().numpy(), rtol=1e-4, atol=1e-5)
    if sys.version_info < (3, 14):
        ts = torch.jit.trace(model, example)
        ts.save(str(run_dir / "model.ts"))
        np.testing.assert_allclose(ref, ts(x).detach().numpy(), rtol=1e-4, atol=1e-5)

    # Everything a serving process needs, next to the weights
    (run_dir / "model_meta.json").write_text(json.dumps({
        "classes": CLASSES, "input_size": [3, 32, 32],
        "mean": CIFAR10_MEAN, "std": CIFAR10_STD,
    }, indent=2))
    log.info("Exported ONNX and torch.export models to %s (parity checked)", run_dir)
