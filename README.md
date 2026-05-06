# Suit Jawa Realtime

Aplikasi Python untuk deteksi gesture Suit Jawa secara real-time menggunakan webcam laptop, MediaPipe Hand Landmarker, dan model MLP Keras.

## Struktur Folder

```text
suit_jawa_realtime/
  app/
    realtime_suit_jawa.py
    realtime_suit_jawa_cnn.py
  docs/
    project_structure.md
  models/
    best_model_cnn_suit_jawa.keras
    model_final_mediapipe_mlp.keras
    hand_landmarker.task
  notebooks/
    suit_jawa_mediapipe_mlp.ipynb
  requirements.txt
  README.md
```

## Install Dependency

```bash
pip install -r requirements.txt
```

## Download Hand Landmarker

Jika file `hand_landmarker.task` belum ada, download dengan perintah:

```bash
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Pastikan File Model Ada

Pastikan file model berikut berada di folder `models/`:

```text
models/model_final_mediapipe_mlp.keras
```

## Cara Menjalankan

Versi MediaPipe landmark + MLP:

```bash
python app/realtime_suit_jawa.py
```

Versi pure CNN dari ROI webcam:

```bash
python app/realtime_suit_jawa_cnn.py
```

## Cara Memakai

- Tunjukkan satu atau dua tangan ke webcam.
- Gesture:
  - gajah = jempol
  - orang = telunjuk
  - semut = kelingking
- Jika dua tangan terdeteksi, sistem menentukan pemenang.
- Tekan `Q` untuk keluar.

Pada script CNN:

- Letakkan gesture di dalam kotak ROI.
- Tekan `1` untuk mode satu ROI.
- Tekan `2` untuk mode dua ROI dan menentukan pemenang dari ROI kiri/kanan.
- Tekan `Q` untuk keluar.

## Aturan Suit Jawa

- gajah menang melawan orang
- orang menang melawan semut
- semut menang melawan gajah
- jika sama maka seri

## Troubleshooting

- Jika kamera tidak terbuka, coba ubah `cv2.VideoCapture(0)` menjadi `cv2.VideoCapture(1)`.
- Jika `hand_landmarker.task` tidak ditemukan, download file task ke folder `models/` terlebih dahulu.
- Jika model tidak ditemukan, pastikan `model_final_mediapipe_mlp.keras` berada di folder `models/`.
- Jika prediksi tidak stabil, pastikan jari terlihat jelas dan tangan tidak terlalu blur.
