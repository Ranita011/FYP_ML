# predict.py

import cv2
import mediapipe as mp
import pickle
import time

# =========================
# Load Model
# =========================

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# MediaPipe Setup
# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Prediction Logic
# =========================

def predict_from_frame(frame):
    """
    Predict gesture from a single frame.
    Returns prediction string.
    """

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Improve performance
    image.flags.writeable = False

    result = hands.process(image)

    image.flags.writeable = True

    if not result.multi_hand_landmarks:
        return "No gesture"

    hand_landmarks = result.multi_hand_landmarks[0]

    data = []

    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y])

    try:
        prediction = model.predict([data])[0]
        return str(prediction)

    except Exception as e:
        print("Prediction Error:", e)
        return "No gesture"


# =========================
# Single Prediction
# =========================

def get_prediction():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return "No camera"

    try:

        # Camera warm-up
        time.sleep(1)

        prediction = "No gesture"

        # Try multiple frames
        for _ in range(20):

            ret, frame = cap.read()

            if not ret:
                continue

            prediction = predict_from_frame(frame)

            if prediction != "No gesture":
                break

        return prediction

    except Exception as e:
        print("Camera Error:", e)
        return "No gesture"

    finally:
        cap.release()


# =========================
# Live Prediction
# =========================

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No camera detected")
        return

    print("Press Q to quit")

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Frame capture failed")
                break

            prediction_text = predict_from_frame(frame)

            # Display prediction
            cv2.putText(
                frame,
                prediction_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Sign Language Detection", frame)

            # Quit on Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Stopped")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# =========================
# Run
# =========================

if __name__ == "__main__":
    main()