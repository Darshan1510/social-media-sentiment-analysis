import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load model and tokenizer
@st.cache_resource
def load_sentiment_model(model_path):
    return load_model(model_path)

@st.cache_resource
def load_tokenizer(tokenizer_path):
    # Load tokenizer from pickle file
    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)
    return tokenizer

# Function to predict sentiment
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

    # File paths and configurations
    models = {
        "CNN Model": "sentiment_analysis_cnn.h5",
        "GRU Model": "sentiment_analysis_gru.h5",
        "LSTM Model": "sentiment_analysis_lstm.h5"
    }
    tokenizer_path = "tokenizer.pkl"
    max_sequence_length = 100

    # User input
    user_input = st.text_area("Enter a Reddit comment for sentiment analysis:")

    # Model selection
    model_choice = st.selectbox("Select a model for prediction:", list(models.keys()))
    model_path = models[model_choice]

    # Load model and tokenizer
    model = load_sentiment_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    # Display model details
    st.info(f"You selected the **{model_choice}** for sentiment analysis.")

    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a valid comment!")
        else:
            sentiment, confidence_scores = predict_sentiment(user_input, model, tokenizer, max_sequence_length)
            st.success(f"The sentiment of the comment is: **{sentiment}**")

            # Display confidence scores
            st.subheader("Confidence Scores:")
            st.bar_chart({
                "Negative": confidence_scores[0],
                "Neutral": confidence_scores[1],
                "Positive": confidence_scores[2],
            })

    # Model comparison analysis
    st.subheader("Model Performance Analysis")
    st.markdown("""
    - **CNN Model**: Fast and efficient for text classification.
    - **GRU Model**: Handles sequential data well and captures context.
    - **LSTM Model**: Excels in understanding long-term dependencies in text.
    """)

if __name__ == "__main__":
    main()
