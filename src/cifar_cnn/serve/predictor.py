import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


class Predictor:
    def __init__(self, model_dir: Path):
        meta = json.loads((model_dir / "model_meta.json").read_text())
        self.classes = meta["classes"]
        self.mean = np.array(meta["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(meta["std"], dtype=np.float32).reshape(3, 1, 1)
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"),
                                            providers=ort.get_available_providers())

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0   # HWC -> CHW, [0, 1]
        return (x - self.mean) / self.std

    def predict(self, images: list[bytes], top_k: int = 3) -> list[dict]:
        batch = np.stack([self.preprocess(b) for b in images])
        logits = self.session.run(None, {"images": batch})[0]
        probs = np.exp(logits - logits.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        results = []
        for p in probs:
            top = np.argsort(-p)[:top_k]
            results.append({
                "label": self.classes[top[0]],
                "confidence": float(p[top[0]]),
                "top_k": [{"label": self.classes[i], "confidence": float(p[i])} for i in top],
            })
        return results