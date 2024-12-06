import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        # Read image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image to get hand landmarks
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [landmark.x for landmark in hand_landmarks.landmark]
                y_ = [landmark.y for landmark in hand_landmarks.landmark]

                # Normalize landmarks to [0, 1] relative to the bounding box
                x_min, x_max = min(x_), max(x_)
                y_min, y_max = min(y_), max(y_)
                normalized_landmarks = [
                    ((landmark.x - x_min) / (x_max - x_min),
                     (landmark.y - y_min) / (y_max - y_min))
                    for landmark in hand_landmarks.landmark
                ]

                # Flatten the normalized landmarks into a single sequence
                flattened_landmarks = [
                    coord for point in normalized_landmarks for coord in point
                ]
                data_aux.append(flattened_landmarks)

            # Take only the first detected hand (for simplicity)
            if data_aux:
                data.append(data_aux[0])  # Append the flattened landmarks
                labels.append(dir_)
        else:
            print(f"No hand detected in: {os.path.join(DATA_DIR, dir_, img_path)}")

# Convert to NumPy array for LSTM
data = np.array(data)  # Shape: (samples, features)
labels = np.array(labels)

# Save data and labels
with open('data_lstm.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
