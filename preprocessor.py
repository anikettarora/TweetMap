import nltk
import re
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download the NLTK stopwords if not already downloaded
nltk.download('stopwords')

stopwords_list = set(stopwords.words('english'))

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    #'''Removes HTML tags: replaces anything between opening and closing <> with empty space'''
    return TAG_RE.sub('', text)


class Preprocess():
    #'''Cleans text data and uses VADER for sentiment analysis'''

    def __init__(self):
        # Initialize VADER sentiment analyzer
        self.analyzer = SentimentIntensityAnalyzer()

    def preprocess_text(self, sen):
        #Preprocesses text: lowercasing, removes tags, punctuation, stopwords
        sen = sen.lower()
        # Remove HTML tags
        sentence = remove_tags(sen)
        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        # Remove extra spaces
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        # Remove stopwords
        pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s*')
        sentence = pattern.sub('', sentence)
        return sentence

    def get_vader_sentiment(self, text):
        #Uses VADER lexicon for sentiment analysis and returns sentiment with confidence
        sentiment_score = self.analyzer.polarity_scores(text)  # Get VADER sentiment scores
        compound_score = sentiment_score['compound']

        # Define sentiment based on compound score
        if compound_score >= 0.05:
            sentiment = 'Positive'
            confidence = compound_score
        elif compound_score <= -0.05:
            sentiment = 'Negative'
            confidence = -compound_score  # Negate for positive confidence value
        else:
            sentiment = 'Neutral'
            confidence = 1 - abs(compound_score)  # Confidence for neutral

        return sentiment, confidence