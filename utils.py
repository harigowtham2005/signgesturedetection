# utils.py

import numpy as np

def flatten_landmarks(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
