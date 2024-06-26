import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model
model = tf.keras.models.load_model('diabetes_detection_model.h5')

# Load the scaler used during training
scaler = StandardScaler()
scaler_filename = 'scaler.save'
scaler = joblib.load(scaler_filename)

# Prepare new input data for prediction
input_data = np.array([[1, 85,  66, 29, 0, 26.6, 51,35,]])
input_data_scaled = scaler.transform(input_data)  # Apply the same scaling

# Perform prediction
predictions = model.predict(input_data_scaled)

# Process the predictions
if predictions[0] >= 0.5:
    result = 'diabetic'
else:
    result = 'non-diabetic'

print('Prediction:',result)