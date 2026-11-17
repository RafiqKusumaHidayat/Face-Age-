import random
import shutil
from pathlib import Path
from collections import Counter

# ==========================================
# KONFIGURASI FOLDER PROJECT
# ==========================================
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "dataset" / "UTKFace"
OUTPUT_DIR = ROOT / "dataset" / "age_yolo"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
random.seed(42)


def classify_age(age: int):
    """Ubah umur numerik UTKFace menjadi salah satu dari lima kelas."""
    if 0 <= age <= 5:
        return "balita"
    if 6 <= age <= 11:
        return "anak_anak"
    if 12 <= age <= 17:
        return "remaja"
    if 18 <= age <= 59:
        return "dewasa"
    if 60 <= age <= 116:
        return "lansia"
    return None


def read_age(image_path: Path):
    """Baca umur dari nama UTKFace: age_gender_race_timestamp.jpg."""
    try:
        parts = image_path.stem.split("_")
        if len(parts) < 4:
            return None

        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])

        if not (0 <= age <= 116):
            return None
        if gender not in (0, 1):
            return None
        if race not in range(5):
            return None

        return age
    except (ValueError, IndexError):
        return None


def main():
    print("=" * 58)
    print("        UTKFACE - PERSIAPAN DATASET USIA")
    print("=" * 58)
    print(f"Lokasi project : {ROOT}")
    print(f"Dataset asli   : {RAW_DIR}")
    print(f"Output         : {OUTPUT_DIR}")

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {RAW_DIR}\n"
            "Pastikan gambar UTKFace berada di dataset/UTKFace."
        )

    classes = ["balita", "anak_anak", "remaja", "dewasa", "lansia"]
    grouped = {name: [] for name in classes}
    skipped = 0
    image_files_seen = 0

    print("\n[1/4] Membaca nama file gambar...")
    for image_path in RAW_DIR.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_files_seen += 1
        age = read_age(image_path)
        if age is None:
            skipped += 1
            continue

        group = classify_age(age)
        if group is None:
            skipped += 1
            continue

        grouped[group].append(image_path)

    total_valid = sum(len(items) for items in grouped.values())
    print(f"File gambar ditemukan : {image_files_seen}")
    print(f"Gambar valid          : {total_valid}")
    print(f"Gambar dilewati       : {skipped}")

    if total_valid == 0:
        raise RuntimeError(
            "Tidak ada gambar UTKFace yang valid. Periksa format nama file."
        )

    print("\n[2/4] Jumlah gambar per kategori:")
    for name in classes:
        count = len(grouped[name])
        print(f"  {name:12}: {count:>6} gambar")
        if count < 3:
            raise RuntimeError(f"Gambar untuk kelas '{name}' kurang dari 3.")

    # Hentikan jika output lama ada; folder sumber UTKFace tidak disentuh.
    print("\n[3/4] Menyiapkan folder train/val/test...")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    splits = ("train", "val", "test")
    for split in splits:
        for name in classes:
            (OUTPUT_DIR / split / name).mkdir(parents=True, exist_ok=True)

    counts = {split: Counter() for split in splits}
    print("[4/4] Menyalin gambar ke pembagian 80:10:10...")

    for name in classes:
        images = grouped[name].copy()
        random.shuffle(images)

        n = len(images)
        n_test = max(1, round(n * 0.10))
        n_val = max(1, round(n * 0.10))

        # Pastikan training tetap memiliki gambar.
        test_images = images[:n_test]
        val_images = images[n_test:n_test + n_val]
        train_images = images[n_test + n_val:]

        split_data = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        for split, image_list in split_data.items():
            for index, source in enumerate(image_list):
                # Tambahkan awalan berurutan untuk mencegah nama file bentrok.
                new_name = f"{name}_{index:06d}_{source.name}"
                destination = OUTPUT_DIR / split / name / new_name
                shutil.copy2(source, destination)
                counts[split][name] += 1

        print(f"  Selesai: {name}")

    print("\n" + "=" * 58)
    print("RINGKASAN HASIL")
    print("=" * 58)
    for split in splits:
        split_total = sum(counts[split].values())
        print(f"\n{split.upper()} ({split_total} gambar)")
        for name in classes:
            print(f"  {name:12}: {counts[split][name]:>6}")

    output_total = sum(sum(c.values()) for c in counts.values())
    print("\n" + "=" * 58)
    print(f"BERHASIL. Total gambar yang disalin: {output_total}")
    print(f"Folder hasil: {OUTPUT_DIR}")
    print("Dataset asli di dataset/UTKFace tidak diubah.")
    print("=" * 58)


if __name__ == "__main__":
    main()
