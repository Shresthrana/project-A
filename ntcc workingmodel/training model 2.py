
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
# Step 1: Load the dataset
dataset_path = r"C:\Users\a_k_r\Desktop\collegematerial\ntcc workingmodel\diabetes.csv"
data = pd.read_csv(dataset_path)

# Step 2: Preprocess the dataset
# Split features and labels
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   #Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
scaler_filename = 'scaler.save'
joblib.dump(scaler, scaler_filename)

# Step 3: Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 4: Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, shuffle= True ,  validation_data=(X_test, y_test))

# Step 6: Evaluate the model
loss, accuracy = ( model.evaluate(X_test, y_test))
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Step 7: Save the trained model
model.save('diabetes_detection_model.h5')
