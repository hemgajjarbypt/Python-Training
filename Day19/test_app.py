import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app import app

client = TestClient(app)

def test_ask_question_success():
    with patch('app.qa_pipeline') as mock_pipeline:
        mock_pipeline.return_value = [{"generated_text": "This is a mock answer."}]
        response = client.post("/qa", json={"question": "What is AI?"})
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is AI?"
        assert data["answer"] == "This is a mock answer."
        mock_pipeline.assert_called_once_with("What is AI?", max_new_tokens=100)

def test_ask_question_empty_question():
    response = client.post("/qa", json={"question": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty"

def test_ask_question_whitespace_only():
    response = client.post("/qa", json={"question": "   "})
    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty"

def test_ask_question_pipeline_exception():
    with patch('app.qa_pipeline') as mock_pipeline:
        mock_pipeline.side_effect = Exception("Pipeline error")
        response = client.post("/qa", json={"question": "Test question"})
        assert response.status_code == 500
        assert response.json()["detail"] == "Pipeline error"

def test_ask_question_missing_question():
    response = client.post("/qa", json={})
    assert response.status_code == 422  # Pydantic validation error

def test_ask_question_invalid_question_type():
    response = client.post("/qa", json={"question": 123})
    assert response.status_code == 422  # Pydantic validation error for int instead of str
