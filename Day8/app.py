from transformers import pipeline

# Load pre-trained sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def main():
    # Analyze text
    sentences = ['The movie was absolutely fantastic, I loved every moment of it!', 'I am really disappointed with the service, it was terrible.', 'The product is okay, not great but not too bad either.']

    for sentence in sentences:
        result = sentiment_pipeline(sentence)
        print(result)

if __name__ == '__main__':
    main()

