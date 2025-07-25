# predict_landmark_realtime.py

import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from config import MODEL_DIR, MODEL_NAME, GESTURE_CLASSES
from utils import flatten_landmarks

model = load_model(f"{MODEL_DIR}/{MODEL_NAME}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            data = flatten_landmarks(hand_landmarks.landmark)
            pred = model.predict(np.array([data]), verbose=0)
            gesture = GESTURE_CLASSES[np.argmax(pred)]

            cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
