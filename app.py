from pathlib import Path
import threading

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO


# ============================================================
# PATHS AND SETTINGS
# ============================================================
ROOT = Path(__file__).resolve().parent
FACE_MODEL_PATH = ROOT / "models" / "yolo26n-face.pt"
AGE_MODEL_PATH = (
    ROOT / "runs" / "classify" / "utkface_age5" / "weights" / "best.pt"
)

FACE_IMAGE_SIZE = 640
FACE_CONFIDENCE = 0.35
AGE_IMAGE_SIZE = 224

# Use the NVIDIA GPU when PyTorch can access CUDA; otherwise use CPU.
DEVICE = 0 if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB per frame

# Prevent simultaneous requests from using the same model objects at once.
inference_lock = threading.Lock()


# ============================================================
# LOAD MODELS ONCE WHEN THE SERVER STARTS
# ============================================================
def load_models():
    missing = []
    if not FACE_MODEL_PATH.is_file():
        missing.append(str(FACE_MODEL_PATH))
    if not AGE_MODEL_PATH.is_file():
        missing.append(str(AGE_MODEL_PATH))

    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "File model belum ditemukan. Pastikan kedua file berikut tersedia:\n"
            + details
        )

    print("=" * 60)
    print("         FACE DETECTION & AGE CLASSIFICATION WEB")
    print("=" * 60)
    print(f"Face model : {FACE_MODEL_PATH}")
    print(f"Age model  : {AGE_MODEL_PATH}")
    print(f"Device     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    face_model = YOLO(str(FACE_MODEL_PATH))
    age_model = YOLO(str(AGE_MODEL_PATH))

    print("Kedua model berhasil dimuat.")
    print("=" * 60)
    return face_model, age_model


face_model, age_model = load_models()


# ============================================================
# LABEL DISPLAY
# ============================================================
DISPLAY_NAMES = {
    "balita": "BALITA (0–5 tahun)",
    "anak_anak": "ANAK-ANAK (6–11 tahun)",
    "anak-anak": "ANAK-ANAK (6–11 tahun)",
    "remaja": "REMAJA (12–17 tahun)",
    "dewasa": "DEWASA (18–59 tahun)",
    "lansia": "LANSIA (60+ tahun)",
}


def get_age_label(class_name: str) -> str:
    normalized = str(class_name).strip().lower().replace(" ", "_")
    return DISPLAY_NAMES.get(normalized, normalized.upper())


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(int(value), upper))


# ============================================================
# ROUTES
# ============================================================
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_available": bool(torch.cuda.is_available()),
        }
    )


@app.post("/detect")
def detect():
    uploaded = request.files.get("frame")
    if uploaded is None:
        return jsonify({"error": "Frame kamera tidak ditemukan."}), 400

    image_bytes = uploaded.read()
    if not image_bytes:
        return jsonify({"error": "Frame kamera kosong."}), 400

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Frame tidak dapat dibaca sebagai gambar."}), 400

    frame_height, frame_width = frame.shape[:2]
    faces = []

    try:
        with inference_lock:
            # Model 1: detect face bounding boxes.
            face_result = face_model.predict(
                source=frame,
                imgsz=FACE_IMAGE_SIZE,
                conf=FACE_CONFIDENCE,
                device=DEVICE,
                verbose=False,
            )[0]

            if face_result.boxes is not None:
                for box in face_result.boxes:
                    x1, y1, x2, y2 = [
                        int(v) for v in box.xyxy[0].detach().cpu().tolist()
                    ]

                    x1 = clamp(x1, 0, frame_width - 1)
                    y1 = clamp(y1, 0, frame_height - 1)
                    x2 = clamp(x2, 0, frame_width)
                    y2 = clamp(y2, 0, frame_height)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Add a small margin so the age classifier sees more than a tight crop.
                    face_w = x2 - x1
                    face_h = y2 - y1
                    pad_x = int(face_w * 0.12)
                    pad_y = int(face_h * 0.15)

                    crop_x1 = clamp(x1 - pad_x, 0, frame_width - 1)
                    crop_y1 = clamp(y1 - pad_y, 0, frame_height - 1)
                    crop_x2 = clamp(x2 + pad_x, 0, frame_width)
                    crop_y2 = clamp(y2 + pad_y, 0, frame_height)

                    face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if face_crop.size == 0 or min(face_crop.shape[:2]) < 32:
                        continue

                    # Model 2: classify the cropped face into one age group.
                    age_result = age_model.predict(
                        source=face_crop,
                        imgsz=AGE_IMAGE_SIZE,
                        device=DEVICE,
                        verbose=False,
                    )[0]

                    if age_result.probs is None:
                        continue

                    class_id = int(age_result.probs.top1)
                    score = float(age_result.probs.top1conf.detach().cpu().item())
                    class_names = age_model.names
                    class_name = class_names[class_id]

                    faces.append(
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "class_name": str(class_name),
                            "label": get_age_label(str(class_name)),
                            "score": round(score * 100, 1),
                        }
                    )

        return jsonify(
            {
                "width": frame_width,
                "height": frame_height,
                "face_count": len(faces),
                "faces": faces,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            }
        )

    except Exception as exc:
        app.logger.exception("Kesalahan saat inferensi")
        return jsonify({"error": f"Gagal memproses frame: {exc}"}), 500


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Ukuran frame terlalu besar. Coba turunkan resolusi kamera."}), 413


if __name__ == "__main__":
    # Local development server. Open http://127.0.0.1:5000 in your browser.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
