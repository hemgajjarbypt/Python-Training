import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app import app, sentiment_pipeline

client = TestClient(app)

@patch('app.sentiment_pipeline')
def test_analyze_success(mock_pipeline):
    mock_pipeline.return_value = [{'label': 'POSITIVE', 'score': 0.9999}]
    response = client.post("/analyze", json={"sentence": "I love this!"})
    assert response.status_code == 200
    assert response.json() == {'label': 'POSITIVE', 'score': 0.9999}

def test_analyze_invalid_json():
    response = client.post("/analyze", data="not json")
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid JSON"}

def test_analyze_missing_sentence():
    response = client.post("/analyze", json={"other": "value"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Missing 'sentence' key"}

def test_analyze_sentence_not_string():
    response = client.post("/analyze", json={"sentence": 123})
    assert response.status_code == 400
    assert response.json() == {"detail": "'sentence' must be a string"}

def test_analyze_sentence_list():
    response = client.post("/analyze", json={"sentence": ["not", "a", "string"]})
    assert response.status_code == 400
    assert response.json() == {"detail": "'sentence' must be a string"}

def test_analyze_sentence_empty_string():
    with patch('app.sentiment_pipeline') as mock_pipeline:
        mock_pipeline.return_value = [{'label': 'NEUTRAL', 'score': 0.5}]
        response = client.post("/analyze", json={"sentence": ""})
        assert response.status_code == 200
        assert response.json() == {'label': 'NEUTRAL', 'score': 0.5}

@patch('app.sentiment_pipeline')
def test_analyze_pipeline_error(mock_pipeline):
    mock_pipeline.side_effect = Exception("Pipeline error")
    response = client.post("/analyze", json={"sentence": "Test sentence"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Error processing sentence"}

def test_analyze_negative_sentiment():
    with patch('app.sentiment_pipeline') as mock_pipeline:
        mock_pipeline.return_value = [{'label': 'NEGATIVE', 'score': 0.8}]
        response = client.post("/analyze", json={"sentence": "I hate this."})
        assert response.status_code == 200
        assert response.json() == {'label': 'NEGATIVE', 'score': 0.8}
