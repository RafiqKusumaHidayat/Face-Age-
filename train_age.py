
from pathlib import Path
import sys
import torch
from ultralytics import YOLO

# ==================================================
# 1. KONFIGURASI PROJECT
# ==================================================

ROOT = Path(__file__).resolve().parent

# Dataset hasil balancing
DATASET_DIR = ROOT / "dataset" / "age_yolo_balanced"

# Folder penyimpanan hasil training
PROJECT_DIR = ROOT / "runs" / "classify"

# Lokasi model pretrained YOLO Classification
MODEL_CANDIDATES = [
    ROOT / "yolo26n-cls.pt",
    ROOT / "models" / "yolo26n-cls.pt"
]

MODEL_PATH = next(
    (path for path in MODEL_CANDIDATES if path.exists()),
    None
)

# Konfigurasi training
EPOCHS = 40
IMAGE_SIZE = 224
BATCH_SIZE = 4
WORKERS = 0

PROJECT_NAME = "utkface_age5"

# Wajib menggunakan GPU
REQUIRE_GPU = True

# ==================================================
# 2. MENAMPILKAN INFORMASI PROJECT
# ==================================================

print("=" * 60)
print("       YOLO FACE AGE CLASSIFICATION")
print("=" * 60)

print(f"\nLokasi project : {ROOT}")
print(f"Dataset        : {DATASET_DIR}")
print(f"Epochs         : {EPOCHS}")
print(f"Image size     : {IMAGE_SIZE}")
print(f"Batch size     : {BATCH_SIZE}")

# ==================================================
# 3. MEMERIKSA DATASET
# ==================================================

print("\n[1] MEMERIKSA DATASET")

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"""
Dataset tidak ditemukan!

Lokasi yang dicari:
{DATASET_DIR}

Jalankan balance_dataset.py terlebih dahulu.
"""
    )

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Folder train tidak ditemukan: {TRAIN_DIR}"
    )

if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Folder val tidak ditemukan: {VAL_DIR}"
    )

EXPECTED_CLASSES = {
    "balita",
    "anak_anak",
    "remaja",
    "dewasa",
    "lansia"
}

train_classes = {
    p.name for p in TRAIN_DIR.iterdir() if p.is_dir()
}

val_classes = {
    p.name for p in VAL_DIR.iterdir() if p.is_dir()
}

if train_classes != EXPECTED_CLASSES:
    raise RuntimeError(
        f"Kategori train tidak sesuai: {train_classes}"
    )

if val_classes != EXPECTED_CLASSES:
    raise RuntimeError(
        f"Kategori validation tidak sesuai: {val_classes}"
    )

print("Dataset berhasil ditemukan!")

# Memeriksa jumlah gambar setiap kelas
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

print("\nDistribusi dataset:")

for category in sorted(EXPECTED_CLASSES):

    train_count = sum(
        1 for p in (TRAIN_DIR / category).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    val_count = sum(
        1 for p in (VAL_DIR / category).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if train_count == 0 or val_count == 0:
        raise RuntimeError(
            f"Data kategori {category} kosong!"
        )

    print(
        f"{category:15} | "
        f"Train: {train_count:5} | "
        f"Val: {val_count:5}"
    )

# ==================================================
# 4. MEMERIKSA MODEL PRETRAINED
# ==================================================

print("\n[2] MEMERIKSA MODEL YOLO")

if MODEL_PATH is None:
    raise FileNotFoundError(
        """
File yolo26n-cls.pt tidak ditemukan.

Letakkan file di salah satu lokasi berikut:

1. Folder utama project:
   yolo26n-cls.pt

2. Folder models:
   models/yolo26n-cls.pt
"""
    )

print(f"Model ditemukan: {MODEL_PATH}")

# ==================================================
# 5. MEMERIKSA GPU NVIDIA
# ==================================================

print("\n[3] MEMERIKSA GPU DAN CUDA")

print("PyTorch version :", torch.__version__)
print("CUDA version    :", torch.version.cuda)

if not torch.cuda.is_available():

    print("\nGPU TIDAK TERDETEKSI!")
    print("Training dibatalkan agar tidak menggunakan CPU.")

    print("\nPeriksa kembali:")
    print("1. PyTorch CUDA sudah terinstal.")
    print("2. Driver NVIDIA sudah aktif.")
    print("3. Virtual environment yang digunakan benar.")

    if REQUIRE_GPU:
        raise RuntimeError(
            "CUDA tidak aktif. Training tidak dilanjutkan."
        )

    DEVICE = "cpu"

else:

    DEVICE = 0

    GPU_NAME = torch.cuda.get_device_name(0)

    GPU_PROPERTIES = torch.cuda.get_device_properties(0)

    GPU_MEMORY = (
        GPU_PROPERTIES.total_memory / (1024 ** 3)
    )

    # Tes operasi CUDA
    test_tensor = torch.rand(
        512, 512, device="cuda"
    )

    torch.cuda.synchronize()

    del test_tensor

    print("\nGPU BERHASIL TERDETEKSI!")
    print(f"Nama GPU       : {GPU_NAME}")
    print(f"VRAM           : {GPU_MEMORY:.2f} GB")
    print("Device         : CUDA:0")
    print("Training       : GPU NVIDIA")

# ==================================================
# 6. MEMUAT MODEL YOLO CLASSIFICATION
# ==================================================

print("\n[4] MEMUAT MODEL YOLO26 CLASSIFICATION")

model = YOLO(str(MODEL_PATH))

print("Model berhasil dimuat!")

# ==================================================
# 7. MEMULAI TRAINING
# ==================================================

print("\n" + "=" * 60)
print("             MEMULAI TRAINING")
print("=" * 60)

print(f"Device    : {DEVICE}")
print(f"GPU       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Epochs    : {EPOCHS}")
print(f"Image Size: {IMAGE_SIZE}")
print(f"Batch     : {BATCH_SIZE}")

try:

    results = model.train(

        # Lokasi dataset
        data=str(DATASET_DIR),

        # Konfigurasi training
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        # Wajib menggunakan GPU
        device=DEVICE,

        # Pengaturan Windows
        workers=WORKERS,

        # Optimasi training
        amp=True,
        cache=False,

        # Menghentikan jika tidak ada peningkatan
        patience=10,

        # Penyimpanan hasil
        project=str(PROJECT_DIR),
        name=PROJECT_NAME,
        exist_ok=True,

        # Reproducibility
        seed=42,
        deterministic=False,

        # Menyimpan grafik dan model
        save=True,
        plots=True,

        # Menampilkan informasi training
        verbose=True

    )

except torch.cuda.OutOfMemoryError:

    print("\nGPU KEHABISAN VRAM!")
    print("Coba turunkan BATCH_SIZE dari 4 menjadi 2.")
    raise

# ==================================================
# 8. MENAMPILKAN HASIL TRAINING
# ==================================================

BEST_MODEL = (
    PROJECT_DIR /
    PROJECT_NAME /
    "weights" /
    "best.pt"
)

LAST_MODEL = (
    PROJECT_DIR /
    PROJECT_NAME /
    "weights" /
    "last.pt"
)

print("\n" + "=" * 60)
print("             TRAINING SELESAI")
print("=" * 60)

if BEST_MODEL.exists():

    print("\nMODEL TERBAIK BERHASIL DISIMPAN!")
    print(f"\nBest Model : {BEST_MODEL}")
    print(f"Last Model : {LAST_MODEL}")

    print("\nModel siap digunakan untuk pengujian.")

else:

    print("\nFile best.pt belum ditemukan.")
    print("Periksa kembali hasil training.")

print("\nLokasi seluruh hasil:")
print(PROJECT_DIR / PROJECT_NAME)

print("\nPROGRAM SELESAI.")
