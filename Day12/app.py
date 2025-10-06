import random
from transformers import pipeline
from datasets import load_dataset

dataset = load_dataset("imdb", split="test")

sentiment_pipeline = pipeline("sentiment-analysis")

def main():
    random_reviews = random.sample(dataset['text'], 5)

    print("Random 5 IMDB reviews with sentiment:\n")

    for i, review in enumerate(random_reviews, 1):
        result = sentiment_pipeline(review[:512])[0]
        print(f"Review {i}: {review}")
        print(f"Sentiment: {result['label']} (score: {result['score']:.4f})")
        print("-" * 60)

if __name__ == '__main__':
    main()