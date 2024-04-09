# Settyl_task

This code implements a machine learning model using TensorFlow and FastAPI for predicting internal status labels based on external status descriptions.

#Model Development
Data Preprocessing: The dataset is loaded from a JSON file. External status descriptions are cleaned and formatted. Labels are one-hot encoded.
Model Development: External status descriptions are converted into numerical representations using tokenization and padding. The model architecture consists of an embedding layer, a flatten layer, and two dense layers. The model is compiled with categorical cross-entropy loss and Adam optimizer.
Model Training and Evaluation: The data is split into training and testing sets. The model is trained for 10 epochs, and its performance is evaluated using accuracy, precision, and recall metrics.
Model Saving: The trained model and tokenizer are saved to disk.

#API Development
API Setup: A FastAPI application is defined.
Input Data: Requests to the API should contain JSON data with the external status description.
Preprocessing: The input data is preprocessed by converting to lowercase and removing non-alphabetic characters.
Prediction: The preprocessed data is tokenized, padded, and used to make predictions using the trained model.
Output: The predicted internal status label is returned as JSON response.
