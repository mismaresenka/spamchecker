# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os # To check for file existence

# --- MODEL TRAINING ---
# This part of the code will only run once, when the app first starts.
# It checks if the model is already trained and saved. If not, it trains and saves it.

def train_model():
    # Load and clean the dataset
    # Make sure 'spam.csv' is in your GitHub repository
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['text'] = df['text'].str.lower()

    # Split data and train the model
    X_train, _, y_train, _ = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Save the trained model and vectorizer
    joblib.dump(model, 'spam_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')

# Check if the model files exist. If not, train the model.
if not os.path.exists('spam_model.joblib') or not os.path.exists('vectorizer.joblib'):
    st.info("Model not found. Training a new model... This may take a moment.")
    train_model()
    st.success("Model trained and saved successfully!")

# --- LOADING THE MODEL ---
# Use st.cache_resource to load the model and vectorizer only once.
# This is faster and more efficient.
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load('spam_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()


# --- APP INTERFACE ---

st.title("📧 Spam Detection App")
st.markdown("Enter a message below to check if it's spam or not.")

user_input = st.text_area("Your message:", "Type here...", height=150)

if st.button("Check Now"):
    if user_input:
        # 1. Preprocess and vectorize the user input
        processed_input = user_input.lower()
        input_vec = vectorizer.transform([processed_input])
        
        # 2. Predict using the loaded model
        prediction = model.predict(input_vec)
        
        # 3. Display the result
        st.subheader("Prediction Result:")
        if prediction[0] == 'spam':
            st.error("This looks like SPAM! 🚨")
        else:
            st.success("This looks like a normal message (HAM). ✅")
    else:
        st.warning("Please enter a message to check.")

st.markdown("---")
st.markdown("Built with Streamlit and Scikit-learn.")