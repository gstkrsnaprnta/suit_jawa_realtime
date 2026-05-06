import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model_cnn_suit_jawa.keras"
CONFIDENCE_THRESHOLD = 70.0

class_names = ["gajah", "orang", "semut"]

model = None
input_height = 160
input_width = 160


def check_required_files():
    if not MODEL_PATH.exists():
        print(f"Error: file model CNN tidak ditemukan: {MODEL_PATH}")
        print("Pastikan best_model_cnn_suit_jawa.keras berada di folder models/.")
        sys.exit(1)


def get_model_input_size(loaded_model):
    input_shape = loaded_model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]

    if height is None or width is None or channels != 3:
        print(f"Error: bentuk input model tidak didukung: {input_shape}")
        print("Script ini mendukung model CNN image input dengan shape (None, height, width, 3).")
        sys.exit(1)

    return int(height), int(width)


def preprocess_roi(roi_bgr):
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (input_width, input_height), interpolation=cv2.INTER_AREA)
    roi_normalized = roi_resized.astype(np.float32) / 255.0
    return np.expand_dims(roi_normalized, axis=0)


def predict_gesture_from_roi(roi_bgr):
    input_data = preprocess_roi(roi_bgr)
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


def make_square_roi(frame, center_x_ratio, center_y_ratio=0.5, size_ratio=0.42):
    height, width, _ = frame.shape
    size = int(min(width, height) * size_ratio)
    center_x = int(width * center_x_ratio)
    center_y = int(height * center_y_ratio)

    x1 = max(center_x - size // 2, 0)
    y1 = max(center_y - size // 2, 0)
    x2 = min(x1 + size, width)
    y2 = min(y1 + size, height)

    x1 = max(x2 - size, 0)
    y1 = max(y2 - size, 0)

    return x1, y1, x2, y2


def draw_text(frame, text, position, color=(255, 255, 255), scale=0.75):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_prediction_box(frame, roi_box, title, gesture, confidence):
    x1, y1, x2, y2 = roi_box
    is_confident = confidence >= CONFIDENCE_THRESHOLD
    label = gesture if is_confident else "Tidak yakin"
    color = (0, 255, 0) if is_confident else (0, 255, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    draw_text(frame, title, (x1, max(y1 - 36, 24)), color)
    draw_text(frame, f"{label} {confidence:.1f}%", (x1, max(y1 - 8, 52)), color)


def draw_panel(frame, lines):
    y = 32
    for text, color in lines:
        draw_text(frame, text, (20, y), color)
        y += 32


def main():
    global model, input_height, input_width

    check_required_files()
    model = tf.keras.models.load_model(str(MODEL_PATH))
    input_height, input_width = get_model_input_size(model)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: kamera tidak bisa dibuka.")
        print("Coba ubah cv2.VideoCapture(0) menjadi cv2.VideoCapture(1).")
        sys.exit(1)

    two_player_mode = False
    window_name = "Suit Jawa CNN Realtime"

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: frame kamera tidak dapat dibaca.")
                break

            frame = cv2.flip(frame, 1)

            if two_player_mode:
                roi_specs = [
                    ("Pemain 1", make_square_roi(frame, 0.28)),
                    ("Pemain 2", make_square_roi(frame, 0.72)),
                ]
            else:
                roi_specs = [("Testing CNN", make_square_roi(frame, 0.5, size_ratio=0.5))]

            predictions = []
            for title, roi_box in roi_specs:
                x1, y1, x2, y2 = roi_box
                roi = frame[y1:y2, x1:x2]
                gesture, confidence, scores = predict_gesture_from_roi(roi)
                predictions.append(
                    {
                        "title": title,
                        "roi_box": roi_box,
                        "gesture": gesture,
                        "confidence": confidence,
                        "scores": scores,
                    }
                )
                draw_prediction_box(frame, roi_box, title, gesture, confidence)

            panel_lines = []
            if two_player_mode:
                player1, player2 = predictions
                p1_label = player1["gesture"] if player1["confidence"] >= CONFIDENCE_THRESHOLD else "Tidak yakin"
                p2_label = player2["gesture"] if player2["confidence"] >= CONFIDENCE_THRESHOLD else "Tidak yakin"

                panel_lines.extend(
                    [
                        (f"Pemain 1: {p1_label} {player1['confidence']:.1f}%", (255, 255, 255)),
                        (f"Pemain 2: {p2_label} {player2['confidence']:.1f}%", (255, 255, 255)),
                    ]
                )

                if (
                    player1["confidence"] < CONFIDENCE_THRESHOLD
                    or player2["confidence"] < CONFIDENCE_THRESHOLD
                ):
                    panel_lines.append(("Gesture belum jelas", (0, 255, 255)))
                else:
                    winner = determine_winner(player1["gesture"], player2["gesture"])
                    panel_lines.append((winner, (0, 255, 0)))
            else:
                prediction = predictions[0]
                label = (
                    prediction["gesture"]
                    if prediction["confidence"] >= CONFIDENCE_THRESHOLD
                    else "Tidak yakin"
                )
                panel_lines.append((f"Prediksi: {label} {prediction['confidence']:.1f}%", (255, 255, 255)))
                panel_lines.append(("Tekan 2 untuk mode dua pemain", (0, 255, 255)))

            panel_lines.append(("Tekan 1: satu ROI | 2: dua ROI | Q: keluar", (200, 200, 200)))
            draw_panel(frame, panel_lines)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("1"):
                two_player_mode = False
            if key == ord("2"):
                two_player_mode = True

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
