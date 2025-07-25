import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Set dataset path
data_dir = "data/Gesture Image Pre-Processed Data"
categories = sorted(os.listdir(data_dir))  # A, B, ..., Z, 0–9
IMG_SIZE = 64

X = []
y = []

# Load and preprocess images
for idx, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    for img_file in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype('float32') / 255.0
            X.append(img)
            y.append(idx)
        except:
            continue

X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
if not os.path.exists("model"):
    os.mkdir("model")
model.save("model/gesture_model.h5")

print("✅ Training complete. Model saved at model/gesture_model.h5")
