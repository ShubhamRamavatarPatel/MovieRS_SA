import streamlit as st
from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = load_model('model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Function to predict sentiment
def predict_sentiment(review):
    try:
        # Convert review to sequence and pad
        sequences = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(sequences, maxlen=200)  # Ensure maxlen matches your model's input

        # Make prediction
        prediction = model.predict(padded_sequence)
        
        # Assuming binary classification with threshold
        sentiment = "positive" if prediction[0] > 0.5 else "negative"
        return sentiment
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit app UI
def main():
    st.title("Movie Review Sentiment Analysis")

    # Input box for the movie review
    review = st.text_area("Enter your movie review:")

    # When button is pressed, predict sentiment
    if st.button("Analyze Sentiment"):
        if review.strip():
            sentiment = predict_sentiment(review)
            st.write(f"Sentiment: {sentiment}")
        else:
            st.write("Please enter a review.")

if __name__ == '__main__':
    main()
