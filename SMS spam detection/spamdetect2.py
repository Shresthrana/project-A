import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import string
from collections import Counter
import pprint
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Reading the data
df_sms = pd.read_csv('spam.csv', encoding='latin-1')
df_sms = df_sms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df_sms = df_sms.rename(columns={"v1": "label", "v2": "sms"})

# Checking values of length of SMS
df_sms['length'] = df_sms['sms'].apply(len)
df_sms['label'] = df_sms['label'].map({'ham': 0, 'spam': 1})
print(df_sms.shape)
print(df_sms.head())

# Exploratory Data Analysis
df_sms['length'].plot(bins=50, kind='hist')
df_sms.hist(column='length', by='label', bins=50, figsize=(10, 4))
plt.show()

# Example preprocessing
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']
lower_case_documents = [d.lower() for d in documents]
sans_punctuation_documents = [d.translate(str.maketrans("", "", string.punctuation)) for d in lower_case_documents]
preprocessed_documents = [[w for w in d.split()] for d in sans_punctuation_documents]
frequency_list = [Counter(d) for d in preprocessed_documents]
pprint.pprint(frequency_list)

# Vectorizing
count_vector = CountVectorizer()
count_vector.fit(documents)
doc_array = count_vector.transform(documents).toarray()
frequency_matrix = pd.DataFrame(doc_array, columns=count_vector.get_feature_names_out())
print(frequency_matrix)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df_sms['sms'], df_sms['label'], test_size=0.20, random_state=1)

# Vectorizing the training and testing data
count_vector = CountVectorizer()
X_train_vectorized = count_vector.fit_transform(X_train)
X_test_vectorized = count_vector.transform(X_test)

# Implementing the Naive Bayes model
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_vectorized, y_train)





# Making predictions
predictions = naive_bayes.predict(X_test_vectorized)

# Evaluating the model
print('Accuracy score: {}'.format(accuracy_score(y_test, predictions)))
print('Precision score: {}'.format(precision_score(y_test, predictions)))
print('Recall score: {}'.format(recall_score(y_test, predictions)))
print('F1 score: {}'.format(f1_score(y_test, predictions)))


model_filename = 'MultinomialNB.pkl'
vectorizer_filename = 'count_vectorizer.pkl'
joblib.dump(naive_bayes, model_filename)
joblib.dump(count_vector, vectorizer_filename)