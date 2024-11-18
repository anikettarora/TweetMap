# Twitter Sentiment Analysis using Transformers and VADER ⌨️🕊️

This project analyzes the sentiment of tweets using a combination of **RoBERTa** (from the Hugging Face Transformers library) for advanced deep learning-based sentiment classification and **VADER** (Valence Aware Dictionary and sEntiment Reasoner) for lexicon-based sentiment analysis. The app allows users to input queries and fetch the top tweets, displaying the predicted sentiment along with confidence scores.

## Features
- Fetches tweets using the Twitter API (Tweepy).
- Analyzes sentiment using **RoBERTa** (transformers-based model) and **VADER** (lexicon-based model).
- Visualizes sentiment distribution using **Streamlit**.
- Handles user queries for real-time sentiment analysis of trending topics.
- Displays sentiment and confidence for each tweet.

## Technologies Used
- **Python**: The primary language used for backend logic.
- **Streamlit**: A fast and interactive UI framework for building web apps.
- **Tweepy**: A Python library for accessing the Twitter API.
- **Transformers**: Hugging Face’s library for pre-trained models (used for RoBERTa).
- **VADER**: A lexicon-based sentiment analysis tool.
- **NLTK**: Used for text preprocessing tasks like stopword removal and tokenization.

## Installation
   ```bash
   pip install -r requirements.txt

   streamlit run app.py

