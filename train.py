import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical

# Load the dataset
data_dict = pickle.load(open('./data_lstm.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Ensure labels are numeric
unique_labels = sorted(set(labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
indexed_labels = np.array([label_to_index[label] for label in labels])

# One-hot encode labels
encoded_labels = to_categorical(indexed_labels)

# Reshape data for LSTM (samples, timesteps, features)
data = data.reshape(data.shape[0], 1, data.shape[1])

# Split dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, shuffle=True, stratify=indexed_labels)

# Define the LSTM model
model = Sequential([
    LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(len(unique_labels), activation='softmax')  # Number of output classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=40, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model using pickle
model_dict = {'model': model, 'label_to_index': label_to_index, 'index_to_label': {v: k for k, v in label_to_index.items()}}
with open('model_lstm.p', 'wb') as f:
    pickle.dump(model_dict, f)
