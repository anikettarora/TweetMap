import tweepy
from preprocessor import Preprocess
from model import predict_sentiment
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file


# Initialize the Preprocess class for text cleaning and VADER sentiment
preprocessor = Preprocess()

Bearer_Token = os.getenv("TWITTER_BEARER_TOKEN")

client = tweepy.Client(bearer_token=Bearer_Token, wait_on_rate_limit=True)

def fetch_tweet(query, language="en"):
    #Fetch tweets from Twitter API based on query

    query_with_lang = f"{query} lang:{language}"

    tweets = client.search_recent_tweets(
        query=query_with_lang,
        tweet_fields=['context_annotations', 'created_at'], 
        max_results=10,
    )
    tweet_array = []
    for tweet in tweets.data:
        tweet_array.append(tweet.text)
    return tweet_array
    

    # except tweepy.errors.TooManyRequests as e:
    #     # If rate limit is exceeded, inform the user
    #     st.write("Rate limit exceeded. Please wait and try again.")
    #     return []

def preprocess_tweets(tweet_array):
    #Preprocesses the tweets (cleaning text)
    processed_tweets = []
    for tweet in tweet_array:
        processed_tweets.append(preprocessor.preprocess_text(tweet))
    return processed_tweets

# Streamlit UI
st.header("Twitter Sentiment Analyst :3")
st.write("This model will predict the sentiments of the topmost tweets fetched by your query.")

query = st.text_input(label="Enter your query to proceed")
if st.button("Submit"):
    tweet_array = fetch_tweet(query=query)
    if tweet_array:
        preprocessed_tweets = preprocess_tweets(tweet_array=tweet_array)

        for i in range(len(preprocessed_tweets)):
            st.header("Tweet:")

            st.write("Original:     ",tweet_array[i])
            st.write("Preprocessed: ",preprocessed_tweets[i])

            # Get sentiment from the pre-trained Hugging Face model
            model_sentiment = predict_sentiment(preprocessed_tweets[i])
            st.header("RoBERTa Sentiment:")
            st.write(model_sentiment)

            # Get sentiment from VADER lexicon
            vader_sentiment = preprocessor.get_vader_sentiment(preprocessed_tweets[i])
            st.header("VADER Sentiment:")
            st.write(vader_sentiment)

            st.write("---------------------------------------------------")
