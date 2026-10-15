import io

import numpy as np
import pytest
import torch
from PIL import Image

from cifar_cnn.export import export_all
from cifar_cnn.models import build_model


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("model")
    export_all(build_model("resnet_small").eval(), path, torch.device("cpu"))
    return path


def _png_bytes(size=(64, 48)):
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (*size[::-1], 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def test_export_writes_all_files(model_dir):
    for name in ("model.onnx", "model.pt2", "model_meta.json"):
        assert (model_dir / name).exists()


def test_predictor_returns_a_label_for_any_image_size(model_dir):
    from cifar_cnn.serve.predictor import Predictor
    results = Predictor(model_dir).predict([_png_bytes(), _png_bytes((300, 200))])
    assert len(results) == 2
    for r in results:
        assert r["label"] in Predictor(model_dir).classes
        assert 0.0 < r["confidence"] <= 1.0
        assert len(r["top_k"]) == 3


def test_api_health_predict_and_input_validation(model_dir, monkeypatch):
    pytest.importorskip("httpx")
    pytest.importorskip("prometheus_fastapi_instrumentator")
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    from fastapi.testclient import TestClient

    from cifar_cnn.serve import app as app_module
    client = TestClient(app_module.app)

    assert client.get("/health").json()["status"] == "ok"
    ok = client.post("/predict", files=[("files", ("a.png", _png_bytes(), "image/png"))])
    assert ok.status_code == 200 and ok.json()["predictions"][0]["label"]
    bad_type = client.post("/predict", files=[("files", ("a.txt", b"hello", "text/plain"))])
    assert bad_type.status_code == 415
    not_an_image = client.post("/predict", files=[("files", ("a.png", b"not an image", "image/png"))])
    assert not_an_image.status_code == 400


def test_predictor_tta_matches_flip_averaged_torch_probs(model_dir):
    from cifar_cnn.serve.predictor import Predictor
    pred = Predictor(model_dir, tta=True)
    png = _png_bytes()
    x = torch.from_numpy(pred.preprocess(png)).unsqueeze(0)
    plain = pred.session.run(None, {"images": x.numpy()})[0]
    flipped = pred.session.run(None, {"images": np.ascontiguousarray(x.numpy()[:, :, :, ::-1])})[0]
    want = (pred._softmax(plain) + pred._softmax(flipped)) / 2
    got = pred.predict([png], top_k=10)[0]["top_k"]
    np.testing.assert_allclose(sorted(g["confidence"] for g in got), sorted(want[0]), rtol=1e-4, atol=1e-5)
