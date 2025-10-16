import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app import app, API_KEY

client = TestClient(app)

# Tests for /sentiment route

@patch('app.sentiment_pipeline')
def test_sentiment_success(mock_pipeline):
    mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.99}]
    response = client.post("/sentiment", json={"sentence": "I love this!"})
    assert response.status_code == 200
    assert response.json() == {"label": "POSITIVE", "score": 0.99}

def test_sentiment_missing_sentence():
    response = client.post("/sentiment", json={})
    assert response.status_code == 400
    assert response.json() == {"detail": "Missing 'sentence' key"}

def test_sentiment_invalid_sentence_type():
    response = client.post("/sentiment", json={"sentence": 123})
    assert response.status_code == 400
    assert response.json() == {"detail": "'sentence' must be a string"}

@patch('app.sentiment_pipeline')
def test_sentiment_pipeline_error(mock_pipeline):
    mock_pipeline.side_effect = Exception("Pipeline error")
    response = client.post("/sentiment", json={"sentence": "Test sentence"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Error processing sentence"}

def test_sentiment_invalid_json():
    response = client.post(
        "/sentiment",
        data='{"sentence": "test"',
        headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]

# Tests for /qa route

@patch('app.qa_pipeline')
def test_qa_success(mock_pipeline):
    mock_pipeline.return_value = [{"generated_text": "The answer is 42."}]
    headers = {"x-api-key": API_KEY}
    response = client.post("/qa", json={"question": "What is the meaning of life?"}, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"question": "What is the meaning of life?", "answer": "The answer is 42."}

def test_qa_empty_question():
    headers = {"x-api-key": API_KEY}
    response = client.post("/qa", json={"question": ""}, headers=headers)
    assert response.status_code == 400
    assert response.json() == {"detail": "Question cannot be empty"}

def test_qa_invalid_api_key():
    headers = {"x-api-key": "wrong-key"}
    response = client.post("/qa", json={"question": "Test question"}, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API Key"}
