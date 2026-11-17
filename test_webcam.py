
import cv2
import torch

from pathlib import Path
from ultralytics import YOLO

# ==========================================
# 1. KONFIGURASI MODEL
# ==========================================

ROOT = Path(__file__).resolve().parent

FACE_MODEL_PATH = (
    ROOT / "models" / "yolo26n-face.pt"
)

AGE_MODEL_PATH = (
    ROOT / "runs" /
    "classify" /
    "utkface_age5" /
    "weights" /
    "best.pt"
)

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. VALIDASI MODEL
# ==========================================

if not FACE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Face model tidak ditemukan: {FACE_MODEL_PATH}"
    )

if not AGE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Age model tidak ditemukan: {AGE_MODEL_PATH}"
    )

print("Device:", DEVICE)

# ==========================================
# 3. MEMUAT KEDUA MODEL
# ==========================================

print("\nMemuat model Face Detection...")

face_model = YOLO(str(FACE_MODEL_PATH))

print("Memuat model Age Classification...")

age_model = YOLO(str(AGE_MODEL_PATH))

print("Kedua model berhasil dimuat!")

# ==========================================
# 4. KATEGORI USIA
# ==========================================

DISPLAY_NAMES = {
    "balita": "BALITA",
    "anak_anak": "ANAK-ANAK",
    "remaja": "REMAJA",
    "dewasa": "DEWASA",
    "lansia": "LANSIA"
}

# ==========================================
# 5. MEMBUKA KAMERA
# ==========================================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not camera.isOpened():
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Kamera tidak berhasil dibuka!"
    )

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nKamera berhasil diaktifkan.")
print("Tekan Q untuk berhenti.")

# ==========================================
# 6. LOOP REAL-TIME
# ==========================================

try:

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Gagal membaca kamera.")
            break

        # Membuat tampilan mirror
        frame = cv2.flip(frame, 1)

        # ----------------------------------
        # FACE DETECTION
        # ----------------------------------

        face_results = face_model.predict(
            source=frame,
            imgsz=640,
            conf=0.35,
            device=DEVICE,
            verbose=False
        )

        face_count = 0

        result = face_results[0]

        if result.boxes is not None:

            for box in result.boxes:

                coords = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(int, coords)

                h, w = frame.shape[:2]

                # Batasi koordinat
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                # Tambahkan margin
                pad_x = int((x2 - x1) * 0.12)
                pad_y = int((y2 - y1) * 0.15)

                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)

                cx2 = min(w, x2 + pad_x)
                cy2 = min(h, y2 + pad_y)

                # Potong wajah
                face_crop = frame[
                    cy1:cy2,
                    cx1:cx2
                ]

                if face_crop.size == 0:
                    continue

                if face_crop.shape[0] < 32:
                    continue

                if face_crop.shape[1] < 32:
                    continue

                # ----------------------------------
                # AGE CLASSIFICATION
                # ----------------------------------

                age_result = age_model.predict(
                    source=face_crop,
                    imgsz=224,
                    device=DEVICE,
                    verbose=False
                )[0]

                if age_result.probs is None:
                    continue

                class_id = int(
                    age_result.probs.top1
                )

                confidence = float(
                    age_result.probs.top1conf.item()
                )

                class_name = str(
                    age_model.names[class_id]
                ).lower()

                age_label = DISPLAY_NAMES.get(
                    class_name,
                    class_name.upper()
                )

                face_count += 1

                # ----------------------------------
                # GAMBAR BOUNDING BOX
                # ----------------------------------

                color = (0, 220, 80)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label = (
                    f"{age_label} "
                    f"({confidence * 100:.1f}%)"
                )

                font = cv2.FONT_HERSHEY_SIMPLEX

                (tw, th), baseline = cv2.getTextSize(
                    label,
                    font,
                    0.6,
                    2
                )

                label_y = max(y1, th + 12)

                cv2.rectangle(
                    frame,
                    (x1, label_y - th - baseline - 8),
                    (x1 + tw + 10, label_y),
                    color,
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, label_y - 5),
                    font,
                    0.6,
                    (255, 255, 255),
                    2
                )

        # ----------------------------------
        # INFORMASI KAMERA
        # ----------------------------------

        cv2.putText(
            frame,
            f"Faces Detected: {face_count}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press Q to Quit",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        # Tampilkan kamera
        cv2.imshow(
            "YOLO FACE AGE DETECTION",
            frame
        )

        # Tekan Q untuk keluar
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("\nKamera berhasil dihentikan.")
