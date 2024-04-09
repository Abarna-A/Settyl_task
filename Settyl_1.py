import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

file_path = "C:/Users/abarn/Downloads/TASK_DATASET/ecf9c1e7ab7374f18e4400b7a3d2a161-f94652f217eeca83e36dab9d08727caf79ebdecf/dataset.json"

# Load the dataset
df = pd.read_json(file_path)

# Data Preprocessing
# Clean and format external status descriptions
df['externalStatus'] = df['externalStatus'].str.replace(r'\(.*\)', '')  # Remove vessel name
df['externalStatus'] = df['externalStatus'].str.replace(r'[^a-zA-Z\s]', '')  # Remove non-alphabetic characters

# One-hot encode the labels
labels = pd.get_dummies(df['internalStatus'])

# Model Development
# Convert external status descriptions to numerical representation
vocab_size = 1000  # Example vocab size
embedding_dim = 16
max_length = 20  # Example max length for padding
trunc_type = 'post'
padding_type = 'post'

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(df['externalStatus'])
sequences = tokenizer.texts_to_sequences(df['externalStatus'])
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Define the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(labels.shape[1], activation='softmax')  # Number of output classes
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model Training and Evaluation
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save("trained_model.h5")

# Save the tokenizer
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w") as json_file:
    json_file.write(tokenizer_json)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(np.array(y_test), axis=1)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
precision = precision_score(y_test_classes, y_pred_classes, average='macro')
recall = recall_score(y_test_classes, y_pred_classes, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
