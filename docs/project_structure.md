# Struktur Project

Project ini dipisahkan berdasarkan fungsi file agar lebih mudah dirawat.

```text
suit_jawa_realtime/
  app/
    realtime_suit_jawa.py
  docs/
    project_structure.md
  models/
    model_final_mediapipe_mlp.keras
    hand_landmarker.task
  notebooks/
    suit_jawa_mediapipe_mlp.ipynb
  requirements.txt
  README.md
```

## Folder

- `app/` berisi kode program utama untuk deteksi gesture real-time dari webcam.
- `models/` berisi model Keras final dan file MediaPipe Hand Landmarker.
- `notebooks/` berisi notebook training model.
- `docs/` berisi dokumentasi tambahan project.

## Menjalankan Aplikasi

Dari root project:

```bash
python app/realtime_suit_jawa.py
```

Jika memakai virtual environment:

```bash
.venv/bin/python app/realtime_suit_jawa.py
```
