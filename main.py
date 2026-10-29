"""YOLO26 Vision App: detection, segmentation, classification and pose in one web UI.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import base64
import os
import subprocess
import tempfile
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent

# Same models as the notebook; override any with an env var (e.g. YOLO_DETECT=yolo26n.pt for speed).
MODEL_NAMES = {
    "detection": os.getenv("YOLO_DETECT", "yolo26s.pt"),
    "segmentation": os.getenv("YOLO_SEGMENT", "yolo26m-seg.pt"),
    "classification": os.getenv("YOLO_CLASSIFY", "yolo26s-cls.pt"),
    "pose": os.getenv("YOLO_POSE", "yolo26s-pose.pt"),
}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))

# Video runs the model on every frame, so default to the fast nano models.
VIDEO_MODEL_NAMES = {
    "detection": os.getenv("YOLO_VIDEO_DETECT", "yolo26n.pt"),
    "segmentation": os.getenv("YOLO_VIDEO_SEGMENT", "yolo26n-seg.pt"),
    "pose": os.getenv("YOLO_VIDEO_POSE", "yolo26n-pose.pt"),
}
MAX_VIDEO_MB = int(os.getenv("MAX_VIDEO_MB", "50"))
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "30"))
VIDEO_DIR = Path(tempfile.gettempdir()) / "yolo26_videos"
VIDEO_DIR.mkdir(exist_ok=True)

app = FastAPI(title="YOLO26 Vision App")

_models: dict = {}
_lock = threading.Lock()
_video_lock = threading.Lock()  # tracker state lives on the model, so one video at a time


def get_model(task: str, video: bool = False):
    """Load a model on first use and cache it (keeps startup fast and memory low)."""
    names = VIDEO_MODEL_NAMES if video else MODEL_NAMES
    key = f"video-{task}" if video else task
    with _lock:
        if key not in _models:
            from ultralytics import YOLO

            _models[key] = YOLO(names[task])
        return _models[key]


def to_data_url(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def summarize(task: str, r) -> dict:
    """Text summary per task, mirroring the notebook's printout."""
    if task == "detection":
        counts = Counter(r.names[int(i)] for i in r.boxes.cls)
        return {"counts": dict(counts), "total": len(r.boxes)}
    if task == "segmentation":
        H, W = r.orig_shape
        items = []
        if r.masks is not None:
            for poly, cid, conf in zip(r.masks.xy, r.boxes.cls, r.boxes.conf):
                area = float(cv2.contourArea(poly.astype(np.float32)))
                items.append(
                    {
                        "label": r.names[int(cid)],
                        "confidence": round(float(conf), 3),
                        "area_px": round(area),
                        "percent_of_image": round(100 * area / (H * W), 1),
                    }
                )
        return {"instances": items, "total": len(items)}
    if task == "classification":
        top5 = [
            {"label": r.names[int(i)], "confidence": round(float(p), 3)}
            for i, p in zip(r.probs.top5, r.probs.top5conf)
        ]
        return {"top5": top5}
    if task == "pose":
        return {"persons": len(r.boxes)}
    return {}


@app.get("/health")
def health():
    return {"status": "ok", "models": MODEL_NAMES, "loaded": sorted(_models)}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    tasks: str = Form("detection,segmentation,classification,pose"),
    conf: float = Form(0.25),
):
    wanted = [t.strip() for t in tasks.split(",") if t.strip()]
    bad = [t for t in wanted if t not in MODEL_NAMES]
    if not wanted or bad:
        raise HTTPException(400, f"Unknown task(s) {bad}. Choose from {list(MODEL_NAMES)}")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read that file as an image")

    out = {}
    for task in wanted:
        try:
            model = get_model(task)
            kwargs = {} if task == "classification" else {"conf": conf}
            r = model(img, verbose=False, **kwargs)[0]
            out[task] = {"image": to_data_url(r.plot()), "summary": summarize(task, r)}
        except Exception as e:  # one failing model should not kill the whole response
            out[task] = {"error": f"{type(e).__name__}: {e}"}
    return JSONResponse({"filename": file.filename, "results": out})


def _cleanup_old_videos(max_age_s: int = 3600) -> None:
    now = time.time()
    for p in VIDEO_DIR.glob("*"):
        if now - p.stat().st_mtime > max_age_s:
            p.unlink(missing_ok=True)


def process_video(src: Path, dst: Path, task: str, track: bool, conf: float) -> dict:
    """Run a model over every frame, write an annotated H.264 mp4, return stats."""
    import imageio_ffmpeg

    cap = cv2.VideoCapture(str(src))
    opened = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps or fps < 1 or fps > 240:
        fps = 25  # some containers don't report a usable frame rate
    if not opened or w < 2 or h < 2 or n_frames < 1:
        raise HTTPException(400, "Could not read that file as a video")
    if n_frames and n_frames / fps > MAX_VIDEO_SECONDS:
        raise HTTPException(413, f"Video too long (max {MAX_VIDEO_SECONDS} s)")

    # Pipe raw frames into ffmpeg so the result is H.264 (plays in every browser).
    # Dimensions must be even for yuv420p.
    ow, oh = w - w % 2, h - h % 2
    ff = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", str(fps),
            "-i", "-", "-an", "-vcodec", "libx264", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
        ],
        stdin=subprocess.PIPE,
    )

    frames = 0
    id_class: dict[int, str] = {}
    max_in_frame = 0
    try:
        with _video_lock:
            model = get_model(task, video=True)
            run = model.track if track else model.predict
            for r in run(source=str(src), stream=True, conf=conf, verbose=False):
                frame = r.plot()[:oh, :ow]
                ff.stdin.write(np.ascontiguousarray(frame).tobytes())
                frames += 1
                if r.boxes is not None:
                    max_in_frame = max(max_in_frame, len(r.boxes))
                    if track and r.boxes.id is not None:
                        for i, c in zip(r.boxes.id, r.boxes.cls):
                            id_class[int(i)] = r.names[int(c)]
    finally:
        ff.stdin.close()
        if ff.wait() != 0:
            raise HTTPException(500, "Video encoding failed")

    stats = {"frames": frames, "fps": round(fps, 1), "max_objects_in_a_frame": max_in_frame}
    if track:
        stats["unique_objects"] = len(id_class)
        stats["unique_by_class"] = dict(Counter(id_class.values()))
    return stats


@app.post("/api/video")
def video(
    file: UploadFile = File(...),
    task: str = Form("detection"),
    track: bool = Form(True),
    conf: float = Form(0.4),
):
    if task not in VIDEO_MODEL_NAMES:
        raise HTTPException(400, f"Unknown task. Choose from {list(VIDEO_MODEL_NAMES)}")
    _cleanup_old_videos()

    vid_id = uuid.uuid4().hex
    src = VIDEO_DIR / f"{vid_id}_in{Path(file.filename or 'v.mp4').suffix or '.mp4'}"
    dst = VIDEO_DIR / f"{vid_id}.mp4"
    size = 0
    with open(src, "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_VIDEO_MB * 1024 * 1024:
                f.close()
                src.unlink(missing_ok=True)
                raise HTTPException(413, f"File too large (max {MAX_VIDEO_MB} MB)")
            f.write(chunk)
    try:
        stats = process_video(src, dst, task, track, conf)
    finally:
        src.unlink(missing_ok=True)
    return {"filename": file.filename, "video_url": f"/api/video/{vid_id}.mp4", "stats": stats}


@app.get("/api/video/{name}")
def get_video(name: str):
    path = VIDEO_DIR / name
    if not name.endswith(".mp4") or "/" in name or not path.is_file():
        raise HTTPException(404, "Video not found (results are kept for 1 hour)")
    return FileResponse(path, media_type="video/mp4", filename=f"result_{name}")


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
