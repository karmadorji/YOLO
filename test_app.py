"""Smoke tests that don't need model weights (models load lazily)."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "YOLO26 Vision App" in r.text


def test_rejects_non_image():
    r = client.post("/api/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_rejects_unknown_task():
    r = client.post(
        "/api/predict",
        files={"file": ("x.jpg", b"x", "image/jpeg")},
        data={"tasks": "bogus"},
    )
    assert r.status_code == 400


def test_video_rejects_unknown_task():
    r = client.post(
        "/api/video", files={"file": ("x.mp4", b"x", "video/mp4")}, data={"task": "bogus"}
    )
    assert r.status_code == 400


def test_video_rejects_non_video():
    r = client.post("/api/video", files={"file": ("x.mp4", b"not a video", "video/mp4")})
    assert r.status_code == 400


def test_video_pipeline_with_fake_model(tmp_path, monkeypatch):
    """Runs the real video/ffmpeg path with a stand-in for the YOLO model."""
    import cv2
    import numpy as np

    import app.main as m

    # Make a tiny 10-frame test video
    src = tmp_path / "in.mp4"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(10):
        vw.write(np.full((48, 64, 3), i * 20, np.uint8))
    vw.release()

    class Boxes:
        id = [1, 2]
        cls = [0, 0]

        def __len__(self):
            return 2

    class Result:
        boxes = Boxes()
        names = {0: "person"}

        def plot(self):
            return np.zeros((48, 64, 3), np.uint8)

    class FakeModel:
        def track(self, **kw):
            return iter([Result() for _ in range(10)])

        predict = track

    monkeypatch.setattr(m, "get_model", lambda task, video=False: FakeModel())
    r = client.post(
        "/api/video",
        files={"file": ("in.mp4", src.read_bytes(), "video/mp4")},
        data={"task": "detection", "track": "true"},
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["stats"]["frames"] == 10
    assert j["stats"]["unique_objects"] == 2
    out = client.get(j["video_url"])
    assert out.status_code == 200 and len(out.content) > 100
