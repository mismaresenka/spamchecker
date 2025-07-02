# SpamChecker

**SpamChecker** is a simple, interactive web app for detecting spam messages using machine learning. Built with Python, Streamlit, and Scikit-learn, it allows users to input any message and instantly find out if it's spam or not.

## Features

- 📨 **Instant Spam Detection:** Enter a message and check if it's spam or ham (not spam).
- ⚡ **Fast & Interactive:** Powered by Streamlit for a responsive web interface.
- 🧠 **Machine Learning Backend:** Uses a Naive Bayes classifier trained on SMS spam data.
- 💾 **Model Persistence:** Loads pre-trained model and vectorizer for quick predictions.

## How It Works

1. The model is trained using a dataset of SMS messages (`spam.csv`) and a Naive Bayes algorithm.
2. User messages are preprocessed and vectorized to match the model's input.
3. The trained model predicts if the message is spam or ham.
4. The result is displayed instantly in the web app.

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages:
  - streamlit
  - scikit-learn
  - pandas
  - joblib

Install dependencies:

```bash
pip install streamlit scikit-learn pandas joblib
```

### Training the Model

Before running the app, train the model using your dataset (`spam.csv`):

```bash
python train_model.py
```
This will generate `spam_model.joblib` and `vectorizer.joblib` files.

### Running the App

```bash
streamlit run app.py
```

The app will open in your web browser. Enter a message to check if it's spam!

## File Overview

- `app.py` &mdash; Streamlit app for spam detection.
- `train_model.py` &mdash; Script to train and save the spam detection model.
- `spam_model.joblib` / `vectorizer.joblib` &mdash; Saved model and vectorizer (generated after training).
- `spam.csv` &mdash; Dataset for training (not included in repo).

## Example

![SpamChecker Screenshot](#) <!-- Add a screenshot if available -->

## License

This project is provided as-is, without a license. Please add a license file if you intend to share or use this code publicly.

---

Built with ❤️ using Streamlit and Scikit-learn.

---

*Note: For more details, see the code in this repository: https://github.com/mismaresenka/spamchecker*
