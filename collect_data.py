import cv2
import mediapipe as mp
import csv
import os

print("STARTING PROGRAM")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

print("Enter label (A/B/C/...): ", end="")
label = input()

if not os.path.exists("data.csv"):
    with open("data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = [label]
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
                row.append(lm.y)

            with open("data.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

    cv2.putText(frame, f"Collecting: {label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()