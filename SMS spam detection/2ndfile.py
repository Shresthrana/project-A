import joblib
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Loading the model and vectorizer from the .pkl file
model_filename = 'MultinomialNB.pkl'
vectorizer_filename = 'count_vectorizer.pkl'
loaded_model = joblib.load(model_filename)
loaded_vectorizer = joblib.load(vectorizer_filename)


# Function to preprocess new SMS messages
def preprocess_sms(sms):
    sms = sms.lower()  # Convert to lowercase
    sms = sms.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return sms

def predict_spam(sms, model, vectorizer):
    """
    Predict if an SMS message is spam or not using the given model and vectorizer.

    Parameters:
    - sms: str, the SMS message to classify.
    - model: the trained machine learning model.
    - vectorizer: the CountVectorizer instance used to transform the SMS message.

    Returns:
    - prediction: array, the predicted label (1 for spam, 0 for not spam).
    - prediction_proba: array, the probabilities for each class (spam and not spam).
    """
    # Preprocess the SMS
    processed_sms = preprocess_sms(sms)

    # Transform the SMS using the vectorizer
    sms_vectorized = vectorizer.transform([processed_sms])

    # Make prediction using the model
    prediction = model.predict(sms_vectorized)

    # Get prediction probabilities
    prediction_proba = model.predict_proba(sms_vectorized)

    return prediction, prediction_proba

def main():
    while True:
        sms = input("Enter a new SMS message (or 'quit' to exit): ")
        if sms.lower() == 'quit':
            break

        prediction, prediction_proba = predict_spam(sms, loaded_model, loaded_vectorizer)
        if prediction[0] == 1:
            print(f"The message is SPAM with a confidence of {prediction_proba[0][1]*100:.2f}%")
        else:
            print(f"The message is NOT SPAM with a confidence of {prediction_proba[0][0]*100:.2f}%")

# Run the main function
if __name__ == "__main__":
    main()

