import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model_final_mediapipe_mlp.keras"
HAND_LANDMARKER_PATH = BASE_DIR / "models" / "hand_landmarker.task"
CONFIDENCE_THRESHOLD = 70.0

class_names = ["gajah", "orang", "semut"]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

model = None


def check_required_files():
    if not MODEL_PATH.exists():
        print(f"Error: file model tidak ditemukan: {MODEL_PATH}")
        print("Pastikan model_final_mediapipe_mlp.keras berada di folder models/.")
        sys.exit(1)

    if not HAND_LANDMARKER_PATH.exists():
        print(f"Error: file MediaPipe Hand Landmarker tidak ditemukan: {HAND_LANDMARKER_PATH}")
        print("Download terlebih dahulu dengan perintah:")
        print(
            "curl -L -o models/hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
        sys.exit(1)


def landmarks_to_features(hand_landmarks):
    """
    Mengubah 21 landmark tangan menjadi 63 fitur ternormalisasi.

    Normalisasi:
    - semua koordinat dikurangi koordinat wrist/titik 0
    - skala dibagi jarak maksimum dari wrist ke titik lain pada bidang x,y
    """
    coords = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks],
        dtype=np.float32,
    )

    wrist = coords[0].copy()
    coords -= wrist

    xy_distances = np.linalg.norm(coords[:, :2], axis=1)
    scale = float(np.max(xy_distances))
    if scale < 1e-6:
        scale = 1.0

    coords /= scale
    return coords.flatten()


def predict_gesture(hand_landmarks):
    features = landmarks_to_features(hand_landmarks)
    input_data = np.expand_dims(features, axis=0)

    predictions = model.predict(input_data, verbose=0)[0]
    class_index = int(np.argmax(predictions))
    gesture = class_names[class_index]
    confidence = float(predictions[class_index] * 100.0)

    return gesture, confidence, predictions


def determine_winner(g1, g2):
    if g1 == g2:
        return "Seri"

    winning_pairs = {
        ("gajah", "orang"),
        ("orang", "semut"),
        ("semut", "gajah"),
    }

    if (g1, g2) in winning_pairs:
        return "Pemain 1 Menang"

    return "Pemain 2 Menang"


def draw_hand(frame, hand_landmarks, label_text):
    height, width, _ = frame.shape

    points = []
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = max(min(xs) - 12, 0), min(max(xs) + 12, width - 1)
    y_min, y_max = max(min(ys) - 12, 0), min(max(ys) + 12, height - 1)

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 180, 0), 2)

    label_y = max(y_min - 10, 24)
    cv2.putText(
        frame,
        label_text,
        (x_min, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_status_panel(frame, lines):
    y = 30
    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 32


def create_hand_landmarker():
    base_options = python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def main():
    global model

    check_required_files()
    model = tf.keras.models.load_model(str(MODEL_PATH))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: kamera tidak bisa dibuka.")
        print("Coba ubah cv2.VideoCapture(0) menjadi cv2.VideoCapture(1).")
        sys.exit(1)

    detector = create_hand_landmarker()
    window_name = "Suit Jawa Realtime"

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: frame kamera tidak dapat dibaca.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)

            result = detector.detect_for_video(mp_image, timestamp_ms)
            detected_hands = result.hand_landmarks

            if not detected_hands:
                draw_status_panel(
                    frame,
                    [("Tangan tidak terdeteksi", (0, 0, 255))],
                )
            else:
                hand_results = []
                for hand_landmarks in detected_hands:
                    mean_x = float(np.mean([landmark.x for landmark in hand_landmarks]))
                    gesture, confidence, scores = predict_gesture(hand_landmarks)
                    display_gesture = gesture if confidence >= CONFIDENCE_THRESHOLD else "Tidak yakin"
                    hand_results.append(
                        {
                            "landmarks": hand_landmarks,
                            "mean_x": mean_x,
                            "gesture": gesture,
                            "display_gesture": display_gesture,
                            "confidence": confidence,
                            "scores": scores,
                        }
                    )

                hand_results.sort(key=lambda item: item["mean_x"])

                for idx, hand_result in enumerate(hand_results, start=1):
                    label_text = (
                        f"P{idx}: {hand_result['display_gesture']} "
                        f"{hand_result['confidence']:.1f}%"
                    )
                    draw_hand(frame, hand_result["landmarks"], label_text)

                status_lines = []
                if len(hand_results) == 1:
                    hand = hand_results[0]
                    status_lines.extend(
                        [
                            (
                                f"Prediksi: {hand['display_gesture']} "
                                f"{hand['confidence']:.1f}%",
                                (255, 255, 255),
                            ),
                            ("Tunjukkan 2 tangan", (0, 255, 255)),
                        ]
                    )
                else:
                    player1, player2 = hand_results[0], hand_results[1]
                    status_lines.extend(
                        [
                            (
                                f"Pemain 1: {player1['display_gesture']} "
                                f"{player1['confidence']:.1f}%",
                                (255, 255, 255),
                            ),
                            (
                                f"Pemain 2: {player2['display_gesture']} "
                                f"{player2['confidence']:.1f}%",
                                (255, 255, 255),
                            ),
                        ]
                    )

                    if (
                        player1["confidence"] < CONFIDENCE_THRESHOLD
                        or player2["confidence"] < CONFIDENCE_THRESHOLD
                    ):
                        status_lines.append(("Gesture belum jelas", (0, 255, 255)))
                    else:
                        winner = determine_winner(player1["gesture"], player2["gesture"])
                        status_lines.append((winner, (0, 255, 0)))

                draw_status_panel(frame, status_lines)

            cv2.putText(
                frame,
                "Tekan Q untuk keluar",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
