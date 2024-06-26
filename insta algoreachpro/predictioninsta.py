
import joblib
import numpy as np
model =joblib.load("instareach.pkl")

# Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
features = np.array([[253.0, 233.0, 50, 9.0, 165.0, 5.0]])
reach=model.predict(features)
print("feartures",reach)