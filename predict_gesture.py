import cv2
import numpy as np
import tensorflow as tf
import os

# Load trained model
model = tf.keras.models.load_model("model/gesture_model.h5")

# Label mapping (ensure it matches folder names from training)
labels = sorted(os.listdir("data/Gesture Image Pre-Processed Data"))

# Start webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Recognition", 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image and define region of interest (ROI)
    frame = cv2.flip(frame, 1)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    roi_resized = cv2.resize(roi, (64, 64))
    roi_normalized = roi_resized.astype('float32') / 255.0
    roi_reshaped = np.expand_dims(roi_normalized, axis=0)

    # Predict gesture
    predictions = model.predict(roi_reshaped)
    predicted_index = np.argmax(predictions[0])
    predicted_label = labels[predicted_index]

    # Display
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Prediction: {predicted_label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
