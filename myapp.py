from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("trained_model.h5")

# Load the tokenizer from JSON file
with open("tokenizer.json", "r") as f:
    tokenizer_json = f.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

# Define the FastAPI app
app = FastAPI()

# Define request body schema
class InputData(BaseModel):
    external_status: str

# Define function to preprocess input data
def preprocess_input_data(text):
    text = text.lower()
    text = text.replace(r'\(.*\)', '')  # Remove vessel name
    text = text.replace(r'[^a-zA-Z\s]', '')  # Remove non-alphabetic characters
    return text

# Define predict route
@app.post("/predict")
def predict(data: InputData):
    # Preprocess input data
    processed_data = preprocess_input_data(data.external_status)
    
    # Tokenize and pad input data
    sequence = tokenizer.texts_to_sequences([processed_data])
    max_length = 20  # Example max length for padding
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Make prediction
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=-1)
    
    return {"predicted_internal_status": predicted_label.item()}
