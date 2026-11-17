
from pathlib import Path
from ultralytics import YOLO
import torch

# ==========================================
# KONFIGURASI
# ==========================================

ROOT = Path(__file__).resolve().parent

DATASET_DIR = (
    ROOT / "dataset" / "age_yolo_balanced"
)

MODEL_PATH = (
    ROOT / "runs" /
    "classify" /
    "utkface_age5" /
    "weights" /
    "best.pt"
)

# ==========================================
# PERIKSA FILE
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

if not (DATASET_DIR / "test").exists():
    raise FileNotFoundError(
        "Folder testing tidak ditemukan."
    )

# ==========================================
# PILIH DEVICE
# ==========================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("=" * 50)
print("       EVALUASI YOLO AGE CLASSIFICATION")
print("=" * 50)

print("Model:", MODEL_PATH)
print("Device:", DEVICE)

# ==========================================
# MEMUAT MODEL
# ==========================================

model = YOLO(str(MODEL_PATH))

# ==========================================
# EVALUASI DATA TESTING
# ==========================================

print("\nMemulai evaluasi data testing...")

metrics = model.val(
    data=str(DATASET_DIR),
    split="test",
    imgsz=224,
    batch=4,
    device=DEVICE,
    plots=True,
    project=str(ROOT / "runs" / "classify"),
    name="test_evaluation",
    exist_ok=True
)

# ==========================================
# MENAMPILKAN HASIL
# ==========================================

print("\n" + "=" * 50)
print("          HASIL EVALUASI MODEL")
print("=" * 50)

print(f"Top-1 Accuracy: {metrics.top1 * 100:.2f}%")
print(f"Top-5 Accuracy: {metrics.top5 * 100:.2f}%")

print("\nLokasi hasil evaluasi:")
print(ROOT / "runs" / "classify" / "test_evaluation")

print("\nEvaluasi selesai!")
