FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Serving needs only ONNX Runtime, not full PyTorch: image is ~300 MB instead of ~2 GB
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/cifar_cnn/serve ./serve
COPY runs/best_model /models/current

RUN useradd --create-home appuser
USER appuser

ENV MODEL_DIR=/models/current MODEL_VERSION=1.0.0
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]