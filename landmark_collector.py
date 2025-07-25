# landmark_collector.py

import os
import cv2
import numpy as np
import mediapipe as mp
from config import DATA_DIR
from utils import flatten_landmarks

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

gesture = input("Enter gesture label (e.g., A, B): ")
gesture_path = os.path.join(DATA_DIR, gesture)
os.makedirs(gesture_path, exist_ok=True)

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            data = flatten_landmarks(hand.landmark)
            np.save(os.path.join(gesture_path, f"{gesture}_{i}.npy"), data)
            i += 1
            print(f"Saved: {gesture}_{i}.npy")

    cv2.imshow("Collecting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
