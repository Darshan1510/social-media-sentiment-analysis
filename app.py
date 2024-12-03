import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os
import requests
from tensorflow.keras.models import load_model

# URLs for Hugging Face-hosted models
MODEL_URLS = {
    "CNN Model": "https://huggingface.co/darshanshah/social_media_sentiment_analysis/resolve/main/sentiment_analysis_cnn.h5",
    "GRU Model": "https://huggingface.co/darshanshah/social_media_sentiment_analysis/resolve/main/sentiment_analysis_gru.h5",
    "LSTM Model": "https://huggingface.co/darshanshah/social_media_sentiment_analysis/resolve/main/sentiment_analysis_lstm.h5"
}

# Tokenizer file path
TOKENIZER_PATH = "tokenizer.pkl"

# Local storage for downloaded models
MODEL_PATHS = {
    "CNN Model": "cnn_model.h5",
    "GRU Model": "gru_model.h5",
    "LSTM Model": "lstm_model.h5"
}

# Function to download model if not already present
def download_model(model_name):
    model_path = MODEL_PATHS[model_name]
    if not os.path.exists(model_path):
        st.info(f"Downloading {model_name}...")
        response = requests.get(MODEL_URLS[model_name], stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success(f"{model_name} downloaded successfully!")
    return model_path

# Load the tokenizer
@st.cache_resource
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)
    return tokenizer

# Load the model
@st.cache_resource
def load_sentiment_model(model_path):
    return load_model(model_path)

# Predict sentiment
def predict_sentiment(user_input, model, tokenizer, max_sequence_length):
    class_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    user_sequences = tokenizer.texts_to_sequences([user_input])
    user_padded = tf.keras.preprocessing.sequence.pad_sequences(
        user_sequences, maxlen=max_sequence_length
    )
    user_predictions = model.predict(user_padded)
    user_pred_classes = np.argmax(user_predictions, axis=1)
    confidence_scores = user_predictions[0]
    return class_mapping[user_pred_classes[0]], confidence_scores

# Streamlit UI
def main():
    st.title("Reddit Comment Sentiment Analysis")
    st.subheader("Analyze the sentiment of Reddit comments using different models!")

    # Model selection
    model_choice = st.selectbox("Select a model for prediction:", list(MODEL_URLS.keys()))

    # Download and load model
    model_path = download_model(model_choice)
    model = load_sentiment_model(model_path)

    # Load tokenizer
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    # User input
    user_input = st.text_area("Enter a Reddit comment for sentiment analysis:")

    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a valid comment!")
        else:
            sentiment, confidence_scores = predict_sentiment(user_input, model, tokenizer, 100)
            st.success(f"The sentiment of the comment is: **{sentiment}**")

            # Display confidence scores
            st.subheader("Confidence Scores:")
            st.bar_chart({
                "Negative": confidence_scores[0],
                "Neutral": confidence_scores[1],
                "Positive": confidence_scores[2],
            })

    # Display model comparison analysis
    st.subheader("Model Performance Comparison")
    st.markdown("""
    - **CNN Model**: Fast and efficient for feature extraction.
    - **GRU Model**: Balances speed and accuracy with sequential memory.
    - **LSTM Model**: Excels in understanding long-term dependencies in text.
    """)

    # Mocked comparison metrics
    st.bar_chart({
        "CNN Model": 82.96,
        "GRU Model": 83.70,
        "LSTM Model": 83.45
    })

if __name__ == "__main__":
    main()
