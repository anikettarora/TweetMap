from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


def predict_sentiment(tweet):
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    labels = ['Negative', 'Neutral', 'Positive']
    
    # Tokenize tweet
    encoded_tweet = tokenizer(tweet, return_tensors='pt')

    # Get model output
    output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    scores = output[0][0].detach().numpy()
    probabilities = softmax(scores)  # Apply softmax for class probabilities
    
    # Determine sentiment and confidence
    max_index = probabilities.argmax()
    sentiment = labels[max_index]
    confidence = probabilities[max_index]  # Confidence is the probability of the predicted sentiment

    return sentiment, confidence