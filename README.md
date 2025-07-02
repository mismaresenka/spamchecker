# 📧 Spam Detection App

A machine learning-powered web application for detecting spam messages using Natural Language Processing techniques. Built with Streamlit and Scikit-learn, this app provides an intuitive interface for real-time spam detection.

## ✨ Features

- **Real-time Spam Detection**: Instantly classify messages as spam or legitimate (ham)
- **Machine Learning Powered**: Uses Multinomial Naive Bayes algorithm for accurate classification
- **Auto-training**: Automatically trains the model when first run if no pre-trained model exists
- **Interactive Web Interface**: Clean and user-friendly Streamlit interface
- **Efficient Caching**: Model and vectorizer are cached for optimal performance
- **Text Preprocessing**: Includes text normalization and vectorization for better accuracy

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mismaresenka/spamchecker.git
   cd spamchecker
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   The app will automatically open in your default browser at `http://localhost:8501`

## 📖 Usage

1. **Enter a message**: Type or paste any text message in the text area
2. **Click "Check Now"**: Press the button to analyze the message
3. **View results**: The app will display whether the message is:
   - ✅ **HAM** (legitimate message)
   - 🚨 **SPAM** (suspicious/unwanted message)

### Example Messages to Try

**Legitimate (HAM):**
- "Hey, are we still meeting for lunch tomorrow?"
- "Thanks for the presentation today, it was very informative."

**Spam:**
- "CONGRATULATIONS! You've won $1000! Click here to claim your prize!"
- "Free entry! Text WIN to 12345 for your chance to win!"

## 🔧 Model Training

The application includes automatic model training functionality:

### First Run
- When you first run the app, it will automatically detect that no trained model exists
- The app will train a new model using the provided `spam.csv` dataset
- Training typically takes a few moments and only happens once
- The trained model and vectorizer are saved as `spam_model.joblib` and `vectorizer.joblib`

### Manual Retraining
To retrain the model with new data:
1. Replace or update the `spam.csv` file with your dataset
2. Delete the existing model files:
   ```bash
   rm spam_model.joblib vectorizer.joblib
   ```
3. Restart the application - it will automatically retrain

### Dataset Format
The `spam.csv` file should contain:
- **Column 1 (v1)**: Labels ('spam' or 'ham')
- **Column 2 (v2)**: Message text
- **Encoding**: latin-1

## 📁 Project Structure

```
spamchecker/
├── app.py              # Main Streamlit application
├── spam.csv            # Training dataset (~5,572 SMS messages)
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── spam_model.joblib  # Trained model (generated after first run)
├── vectorizer.joblib  # Text vectorizer (generated after first run)
└── .devcontainer/     # Development container configuration
```

### Main Files Description

- **`app.py`**: The core application containing:
  - Model training function
  - Model loading and caching
  - Streamlit user interface
  - Text preprocessing and prediction logic

- **`spam.csv`**: SMS Spam Collection dataset containing labeled examples of spam and legitimate messages

- **`requirements.txt`**: Lists all Python dependencies including Streamlit, Scikit-learn, Pandas, and Joblib

- **Model files** (generated automatically):
  - `spam_model.joblib`: Serialized Multinomial Naive Bayes model
  - `vectorizer.joblib`: Serialized CountVectorizer for text preprocessing

## 🛠️ Technologies Used

- **[Python](https://python.org/)**: Core programming language
- **[Streamlit](https://streamlit.io/)**: Web application framework for the user interface
- **[Scikit-learn](https://scikit-learn.org/)**: Machine learning library for model training and prediction
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis
- **[Joblib](https://joblib.readthedocs.io/)**: Efficient serialization of Python objects (model persistence)

### Machine Learning Details

- **Algorithm**: Multinomial Naive Bayes
- **Feature Extraction**: CountVectorizer (Bag of Words)
- **Text Preprocessing**: Lowercase conversion
- **Train/Test Split**: 70/30 split for model evaluation
- **Dataset Size**: ~5,572 labeled SMS messages

## 🔍 How It Works

1. **Data Loading**: Loads the SMS dataset from `spam.csv`
2. **Preprocessing**: Converts text to lowercase and removes unnecessary columns
3. **Feature Extraction**: Uses CountVectorizer to convert text into numerical features
4. **Model Training**: Trains a Multinomial Naive Bayes classifier
5. **Model Persistence**: Saves the trained model and vectorizer using Joblib
6. **Prediction**: For new messages:
   - Preprocesses the input text
   - Vectorizes using the saved vectorizer
   - Predicts using the trained model
   - Returns classification result

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Built with ❤️ using Streamlit and Scikit-learn**