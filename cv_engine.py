# -*- coding: utf-8 -*-
from pathlib import Path
import urllib.request
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

YUNET = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx?download=true",
]
SFACE_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx?download=true",
]

def _download(urls, dst: Path, min_bytes: int):
    if dst.exists() and dst.stat().st_size >= min_bytes:
        return
    last_err = None
    for u in urls:
        try:
            print(f"Downloading model -> {dst.name}\n  from: {u}")
            urllib.request.urlretrieve(u, dst)
            if dst.stat().st_size < min_bytes:
                raise RuntimeError(f"Downloaded file too small ({dst.stat().st_size} bytes).")
            return
        except Exception as e:
            last_err = e
            try:
                if dst.exists():
                    dst.unlink()
            except Exception:
                pass
    raise RuntimeError(f"Failed to download {dst.name}. Last error: {last_err}")

def ensure_models():
    _download(YUNET_URLS, YUNET, min_bytes=200_000)
    _download(SFACE_URLS, SFACE, min_bytes=10_000_000)

def make_detector(score_thresh=0.9, nms_thresh=0.3, top_k=5000):
    ensure_models()
    detector = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), float(score_thresh), float(nms_thresh), int(top_k))
    return detector

def make_recognizer():
    ensure_models()
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE), "")
    return recognizer

def detect_faces(detector, bgr: np.ndarray):
    h, w = bgr.shape[:2]
    detector.setInputSize((w, h))
    faces = detector.detect(bgr)
    return faces[1]

def pick_largest_face(faces_mat):
    if faces_mat is None or len(faces_mat) == 0:
        return None
    areas = (faces_mat[:,2] * faces_mat[:,3]).astype(np.float32)
    idx = int(np.argmax(areas))
    return faces_mat[idx]

def extract_feature(detector, recognizer, bgr: np.ndarray):
    faces_mat = detect_faces(detector, bgr)
    face = pick_largest_face(faces_mat)
    if face is None:
        return None, None, "NO_FACE"
    aligned = recognizer.alignCrop(bgr, face)
    feat = recognizer.feature(aligned)
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    return feat, face, "OK"

def cosine_sim(recognizer, f1: np.ndarray, f2: np.ndarray) -> float:
    return float(recognizer.match(f1.reshape(1,-1), f2.reshape(1,-1), cv2.FaceRecognizerSF_FR_COSINE))

def draw_face_box(bgr, face_row, color=(0,255,0)):
    if face_row is None:
        return bgr
    x, y, w, h = [int(v) for v in face_row[:4]]
    out = bgr.copy()
    cv2.rectangle(out, (x,y), (x+w, y+h), color, 2)
    return out
