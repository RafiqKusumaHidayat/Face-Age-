
import random
import shutil
from pathlib import Path
from collections import Counter

# ==========================================
# KONFIGURASI
# ==========================================

ROOT = Path(__file__).resolve().parent

SOURCE_DIR = ROOT / "dataset" / "age_yolo"
OUTPUT_DIR = ROOT / "dataset" / "age_yolo_balanced"

CLASSES = [
    "balita",
    "anak_anak",
    "remaja",
    "dewasa",
    "lansia"
]

SPLITS = ["train", "val", "test"]

TARGET_PER_CLASS = 4000

EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(42)


# ==========================================
# FUNGSI MENGAMBIL GAMBAR
# ==========================================

def get_images(folder):

    return [
        path for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in EXTENSIONS
    ]


# ==========================================
# PROGRAM UTAMA
# ==========================================

def main():

    print("=" * 55)
    print("          UTKFACE DATA BALANCING")
    print("=" * 55)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset sumber tidak ditemukan: {SOURCE_DIR}"
        )

    # Hapus hasil balancing lama saja
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    # ======================================
    # PROSES TRAIN
    # ======================================

    print("\n[1] Menyeimbangkan data training...")

    train_counts = Counter()

    for category in CLASSES:

        source_folder = SOURCE_DIR / "train" / category
        destination_folder = OUTPUT_DIR / "train" / category

        destination_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        images = get_images(source_folder)

        if not images:
            raise RuntimeError(
                f"Gambar tidak ditemukan: {source_folder}"
            )

        # Jika jumlah lebih dari target:
        # Ambil sampel secara acak.
        if len(images) >= TARGET_PER_CLASS:

            selected = random.sample(
                images,
                TARGET_PER_CLASS
            )

        else:
            # Jika jumlah kurang:
            # Pertahankan seluruh gambar asli,
            # kemudian lakukan oversampling.
            selected = list(images)

            selected.extend(
                random.choices(
                    images,
                    k=TARGET_PER_CLASS - len(images)
                )
            )

        random.shuffle(selected)

        # Salin gambar dengan nama unik
        for index, image in enumerate(selected):

            new_name = (
                f"{category}_{index:06d}_{image.name}"
            )

            destination = destination_folder / new_name

            shutil.copy2(image, destination)

            train_counts[category] += 1

        print(
            f"{category:15}: "
            f"{len(images)} -> "
            f"{train_counts[category]} gambar"
        )

    # ======================================
    # SALIN VALIDATION DAN TEST
    # TANPA BALANCING
    # ======================================

    print("\n[2] Menyalin validation dan testing...")

    for split in ["val", "test"]:

        source_split = SOURCE_DIR / split
        destination_split = OUTPUT_DIR / split

        if not source_split.exists():
            raise FileNotFoundError(
                f"Folder {split} tidak ditemukan."
            )

        shutil.copytree(
            source_split,
            destination_split
        )

        print(f"Berhasil menyalin {split.upper()}")

    # ======================================
    # LAPORAN HASIL
    # ======================================

    print("\n" + "=" * 55)
    print("             HASIL DATA BALANCING")
    print("=" * 55)

    for category in CLASSES:
        print(
            f"{category:15}: "
            f"{train_counts[category]} gambar"
        )

    print(f"\nTotal training: {sum(train_counts.values())}")

    print("\nLokasi dataset baru:")
    print(OUTPUT_DIR)

    print("\nDATA BALANCING BERHASIL!")


if __name__ == "__main__":
    main()
