# Suit Jawa Realtime

Aplikasi Python untuk deteksi gesture Suit Jawa secara real-time menggunakan webcam laptop, MediaPipe Hand Landmarker, dan model MLP Keras.

## Struktur Folder

```text
suit_jawa_realtime/
  model_final_mediapipe_mlp.keras
  hand_landmarker.task
  realtime_suit_jawa.py
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
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Pastikan File Model Ada

Pastikan file model berikut berada di folder yang sama dengan program:

```text
model_final_mediapipe_mlp.keras
```

## Cara Menjalankan

```bash
python realtime_suit_jawa.py
```

## Cara Memakai

- Tunjukkan satu atau dua tangan ke webcam.
- Gesture:
  - gajah = jempol
  - orang = telunjuk
  - semut = kelingking
- Jika dua tangan terdeteksi, sistem menentukan pemenang.
- Tekan `Q` untuk keluar.

## Aturan Suit Jawa

- gajah menang melawan orang
- orang menang melawan semut
- semut menang melawan gajah
- jika sama maka seri

## Troubleshooting

- Jika kamera tidak terbuka, coba ubah `cv2.VideoCapture(0)` menjadi `cv2.VideoCapture(1)`.
- Jika `hand_landmarker.task` tidak ditemukan, download file task terlebih dahulu.
- Jika model tidak ditemukan, pastikan `model_final_mediapipe_mlp.keras` berada di folder yang sama.
- Jika prediksi tidak stabil, pastikan jari terlihat jelas dan tangan tidak terlalu blur.
