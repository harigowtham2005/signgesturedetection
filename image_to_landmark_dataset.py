# image_to_landmark_dataset.py

import os
import cv2
import mediapipe as mp
import numpy as np
from utils import flatten_landmarks
from config import DATA_DIR

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

for gesture in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, gesture)
    if not os.path.isdir(folder_path):
        continue

    images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for idx, img_name in enumerate(images):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping {img_path} (not a valid image)")
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_img)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            data = flatten_landmarks(hand_landmarks.landmark)
            np.save(os.path.join(folder_path, f"{gesture}_{idx}.npy"), data)
            print(f"Saved: {gesture}_{idx}.npy")
        else:
            print(f"No hand detected in {img_path}")

print("✔ All image landmarks saved as .npy files")
