import pickle
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import requests

# Load the LSTM model and label mappings
model_dict = pickle.load(open('./model_lstm.p', 'rb'))
model = model_dict['model']
index_to_label = model_dict['index_to_label']

# Initialize Mediapipe Hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BaseURL = "http://192.168.8.28:80/"
d_dictionary = {1: 'led1', 2: 'led2', 3: 'led3', 4: 'led4', 5: 'led5'}
response = -1
temp = -1
labels_dict = {0: 'A', 1: 'B', 2: 'L', 3: 'V', 4: 'W'}


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Reshape input for the LSTM model
        data_aux = np.asarray(data_aux).reshape(1, 1, -1)

        # Predict with the LSTM model
        prediction = model.predict(data_aux)
        predicted_label = labels_dict[index_to_label[np.argmax(prediction)]]

        # send request to micro to do something
        currentState = int(prediction[0]) + 1
        if currentState != int(temp) & int(temp) != -1:
            response = requests.get(url=BaseURL + d_dictionary[temp]).text

        if int(response) != currentState:
            response = requests.get(url=BaseURL + d_dictionary[currentState]).text
            if int(response) != temp:
                temp = int(response)

        # Draw prediction on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('Sign Language Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
