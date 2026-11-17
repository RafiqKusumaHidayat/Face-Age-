FACEAGE AI - INTEGRASI WEB FACE DETECTION + AGE GROUP CLASSIFICATION

1. Salin app.py ke folder utama project "Project Deteksi Umur".
2. Salin folder templates dan static ke folder utama project.
3. Pastikan dua model tersedia di lokasi berikut:
   models/yolo26n-face.pt
   runs/classify/utkface_age5/weights/best.pt
4. Aktifkan virtual environment:
   .\.venv\Scripts\Activate.ps1
5. Install Flask saja (jangan install ulang torch jika CUDA sudah berhasil):
   python -m pip install -r requirements-web.txt
6. Jalankan server:
   python app.py
7. Buka browser:
   http://127.0.0.1:5000
8. Klik "Mulai Kamera" dan izinkan akses kamera.

Catatan:
- Kamera browser mengirim frame ke Flask di komputer lokal.
- Frame hanya diproses di memori; aplikasi ini tidak menyimpan foto/video.
- Untuk akses kamera, gunakan localhost/127.0.0.1.
- Jika model face detector tidak kompatibel dengan versi Ultralytics terpasang,
  periksa error saat app.py memuat models/yolo26n-face.pt.
- Hasil usia adalah perkiraan kelompok usia dari penampilan wajah, bukan verifikasi usia.
- Skor klasifikasi adalah confidence keluaran model dan belum tentu terkalibrasi.
