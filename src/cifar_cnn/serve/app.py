import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator

from .predictor import Predictor

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models/current"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")
MAX_BYTES = 5 * 1024 * 1024
ALLOWED = {"image/png", "image/jpeg", "image/webp"}
LOW_CONFIDENCE = 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("serve")
app = FastAPI(title="CIFAR-10 classifier", version=MODEL_VERSION)
Instrumentator().instrument(app).expose(app)          # /metrics for Prometheus
predictor = Predictor(MODEL_DIR)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    if len(files) > 32:
        raise HTTPException(413, "At most 32 images per request")
    blobs = []
    for f in files:
        if f.content_type not in ALLOWED:
            raise HTTPException(415, f"{f.filename}: unsupported type {f.content_type}")
        data = await f.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{f.filename}: larger than 5 MB")
        blobs.append(data)

    t0 = time.perf_counter()
    try:
        results = predictor.predict(blobs)
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(400, f"Could not process image: {e}") from e
    latency_ms = (time.perf_counter() - t0) * 1000

    for r in results:
        r["low_confidence"] = r["confidence"] < LOW_CONFIDENCE    # route to human review
        log.info("prediction label=%s conf=%.3f latency_ms=%.1f version=%s",
                 r["label"], r["confidence"], latency_ms, MODEL_VERSION)
    return {"model_version": MODEL_VERSION, "latency_ms": round(latency_ms, 2), "predictions": results}