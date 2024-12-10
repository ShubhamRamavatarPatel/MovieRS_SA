import streamlit as st
import pickle
import requests
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import warnings
import logging
import os
import streamlit.components.v1 as components
from bs4 import BeautifulSoup

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Load trained sentiment analysis model and tokenizer
model = load_model('model.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the preprocess text function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# Define the function to analyze sentiment
def analyze_sentiment(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment

# Define the function to fetch IMDb ID using OMDb API
def fetch_imdb_id(movie_title, api_key='49aba306'):
    search_url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    response = requests.get(search_url).json()
    imdb_id = response.get('imdbID')
    return imdb_id

# Define the function to fetch movie reviews from IMDb
def fetch_reviews(imdb_id):
    reviews_url = f"https://www.imdb.com/title/{imdb_id}/reviews"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    }
    response = requests.get(reviews_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    reviews = soup.find_all('div', class_='ipc-html-content-inner-div', limit=1)  # Analyze one review for simplicity
    if not reviews:
        return 'No reviews available'
    
    review = reviews[0].get_text()
    return analyze_sentiment(review)

# Define the function to fetch movie posters using OMDb API
def fetch_poster(movie_title, api_key='49aba306'):
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    data = requests.get(url).json()
    poster_url = data.get('Poster')
    if poster_url and poster_url != 'N/A':
        return poster_url
    return "https://via.placeholder.com/500x750?text=No+Poster+Available"

# Load movie data and similarity matrix
movies = pickle.load(open("movies_list2.pkl", 'rb'))
similarity = pickle.load(open("similarity.pkl", 'rb'))
movies_list = movies['title'].values

# Define the recommendation function
def get_recommendations(title, num_recommendations=5):
    try:
        idx = movies[movies['title'] == title].index[0]
    except IndexError:
        st.error(f"Movie '{title}' not found in the dataset.")
        return [], [], []

    movie_language = movies['original_language'].iloc[idx]
    movie_id = movies['id'].iloc[idx]

    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = [s for s in sim_scores if movies['original_language'].iloc[s[0]] == movie_language]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:101]  # Larger pool for randomness

    movie_indices = [i[0] for i in sim_scores]
    random.shuffle(movie_indices)

    if len(movie_indices) > num_recommendations:
        movie_indices = random.sample(movie_indices, num_recommendations)
    else:
        num_recommendations = len(movie_indices)

    selected_titles = movies['title'].iloc[movie_indices].tolist()
    selected_posters = [fetch_poster(title) for title in selected_titles]
    selected_imdb_ids = [fetch_imdb_id(title) for title in selected_titles]
    selected_sentiments = [fetch_reviews(imdb_id) for imdb_id in selected_imdb_ids if imdb_id]  # Check if imdb_id is not None

    return selected_titles, selected_posters, selected_sentiments

# Streamlit app layout
st.header("Movie Recommender System")

# Example movie IDs for the carousel
imageCarouselComponent = components.declare_component("image-carousel-component", path="public")
imageUrls = [
    fetch_poster("Iron Man"),  # Example movie titles
    fetch_poster("Avengers"),
    fetch_poster("Dilwale Dulhania Le Jayenge"),
    fetch_poster("The Dark Knight"),
    fetch_poster("A Wednesday"),
    fetch_poster("The Matrix"),
    fetch_poster("Barfi!"),
    fetch_poster("Pulp Fiction"),
    fetch_poster("Forrest Gump"),
    fetch_poster("The Godfather")
]
imageCarouselComponent(imageUrls=imageUrls, height=200)

selectvalue = st.selectbox("Select movie from dropdown", movies_list)

if st.button("Show Recommend"):
    movie_names, movie_posters, movie_sentiments = get_recommendations(selectvalue)
    cols = st.columns(5)
    
    for i in range(len(cols)):
        with cols[i]:
            if i < len(movie_names):
                st.text(movie_names[i])
                st.image(movie_posters[i])
                st.text(movie_sentiments[i])
