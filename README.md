# YOLO26 Vision App

Web interface for the notebook's pipeline: upload an image and get **detection, segmentation, classification and pose** results. Served on **port 8000**.

## Run locally
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or normal torch for GPU
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000. Weights download on the first request.

## Run with Docker
```bash
docker build -t yolo26-app .
docker run -p 8000:8000 -v $PWD/weights:/srv/weights yolo26-app
```

## GitHub workflow
`.github/workflows/ci.yml` runs on every push/PR: lint, tests, builds the image, starts it on port 8000 and checks `/health`. On pushes to `main` it publishes the image to `ghcr.io/<owner>/<repo>:latest`.
Deploy anywhere with: `docker run -p 8000:8000 ghcr.io/<owner>/<repo>:latest`

## Config (env vars)
`YOLO_DETECT`, `YOLO_SEGMENT`, `YOLO_CLASSIFY`, `YOLO_POSE` (weights files; use `yolo26n*.pt` for speed), `MAX_UPLOAD_MB`.
Video: `YOLO_VIDEO_DETECT`, `YOLO_VIDEO_SEGMENT`, `YOLO_VIDEO_POSE` (default nano models), `MAX_VIDEO_MB` (50), `MAX_VIDEO_SECONDS` (30).

## API
`POST /api/predict` (multipart: `file`, `tasks`, `conf`) · `POST /api/video` (`file`, `task`, `track`, `conf`) · `GET /health`
