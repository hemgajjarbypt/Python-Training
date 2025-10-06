import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app import app

client = TestClient(app)

# Mock data for pipelines
mock_sentiment_result = [{"label": "POSITIVE", "score": 0.99}]
mock_summary_result = [{"summary_text": "This is a summary."}]

def test_sentiment_valid():
    response = client.post("/sentiment", json={"sentence": "I love this product!"})
    assert response.status_code == 200
    assert "label" in response.json()
    assert "score" in response.json()

def test_sentiment_invalid_json():
    response = client.post("/sentiment", data="not a json")
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid JSON"}

def test_sentiment_missing_sentence():
    response = client.post("/sentiment", json={"text": "missing sentence key"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Missing 'sentence' key"}

def test_sentiment_sentence_not_string():
    response = client.post("/sentiment", json={"sentence": 123})
    assert response.status_code == 400
    assert response.json() == {"detail": "'sentence' must be a string"}

@patch("app.sentiment_pipeline")
def test_sentiment_pipeline_error(mock_pipeline):
    mock_pipeline.side_effect = Exception("Pipeline error")
    response = client.post("/sentiment", json={"sentence": "test"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Error processing sentence"}

def test_summary_valid():
    response = client.post("/summary", json={"sentence": "This is a long text that needs to be summarized."})
    assert response.status_code == 200
    assert "summary_text" in response.json()

def test_summary_invalid_json():
    response = client.post("/summary", data="not a json")
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid JSON"}

def test_summary_missing_sentence():
    response = client.post("/summary", json={"text": "missing sentence key"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Missing 'sentence' key"}

def test_summary_sentence_not_string():
    response = client.post("/summary", json={"sentence": 123})
    assert response.status_code == 400
    assert response.json() == {"detail": "'sentence' must be a string"}

@patch("app.summarizer")
def test_summary_pipeline_error(mock_summarizer):
    mock_summarizer.side_effect = Exception("Pipeline error")
    response = client.post("/summary", json={"sentence": "test"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Error processing sentence"}
