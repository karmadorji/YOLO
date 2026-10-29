FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /srv

# libgl/glib are needed by OpenCV at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image far smaller; remove this line and the index URL for GPU hosts
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Weights are downloaded on first use; mount a volume at /srv/weights to persist them.
ENV YOLO_DETECT=weights/yolo26s.pt \
    YOLO_SEGMENT=weights/yolo26m-seg.pt \
    YOLO_CLASSIFY=weights/yolo26s-cls.pt \
    YOLO_POSE=weights/yolo26s-pose.pt
RUN mkdir -p /srv/weights

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
