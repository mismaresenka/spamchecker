# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib # Used for saving the model and vectorizer

print("Starting the training process...")

# Load and clean the dataset
# Make sure you have the 'spam.csv' file in the same folder
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
print("Dataset loaded and cleaned.")

# Simple normalization
df['text'] = df['text'].str.lower()

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)
print("Data split into training and testing sets.")

# Convert text to numeric features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
print("Text data vectorized.")

# Train a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("Model training complete.")

# --- SAVING THE MODEL AND VECTORIZER ---
# These files will be loaded by our Streamlit app
joblib.dump(model, 'spam_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("Model and vectorizer have been saved as 'spam_model.joblib' and 'vectorizer.joblib'")