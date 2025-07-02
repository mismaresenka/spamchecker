# app.py

import streamlit as st
import joblib
import pandas as pd # Required for the vectorizer

# --- LOAD THE SAVED MODEL AND VECTORIZER ---
# We use st.cache_data to load the model only once, which makes the app faster.
@st.cache_data
def load_model_and_vectorizer():
    model = joblib.load('spam_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()


# --- APP INTERFACE ---

# Title for the app
st.title("📧 Spam Detection App")

# A subheader
st.markdown("Enter a message below to check if it's spam or not.")

# Text area for user input
user_input = st.text_area("Your message:", "Type here...", height=150)

# Prediction button
if st.button("Check Now"):
    if user_input:
        # 1. Preprocess the user input (lowercase)
        processed_input = user_input.lower()
        
        # 2. Vectorize the input using the loaded vectorizer
        input_vec = vectorizer.transform([processed_input])
        
        # 3. Predict using the loaded model
        prediction = model.predict(input_vec)
        
        # 4. Display the result
        st.subheader("Prediction Result:")
        if prediction[0] == 'spam':
            st.error("This looks like SPAM! 🚨")
        else:
            st.success("This looks like a normal message (HAM). ✅")
    else:
        st.warning("Please enter a message to check.")

# Add a footer
st.markdown("---")
st.markdown("Built with Streamlit and Scikit-learn.")