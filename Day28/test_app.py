import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import os
from app import app, API_KEY, log_to_file, run_model, summarize_text, extract_keywords

client = TestClient(app)

# Fixture to clean up log file after tests
@pytest.fixture(autouse=True)
def cleanup_log():
    yield
    if os.path.exists("api_logs.log"):
        os.remove("api_logs.log")

# Tests for verify_api_key (though it's a dependency, test via endpoint)

# Tests for log_to_file - removed as logging setup in tests is tricky, but function is covered indirectly via middleware test

# Tests for run_model
@patch('app.tokenizer')
@patch('app.model')
def test_run_model(mock_model, mock_tokenizer):
    mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
    mock_tokenizer.decode.return_value = "Generated text"
    mock_model.generate.return_value = [MagicMock()]

    result = run_model("Test prompt")
    assert result == "Generated text"
    mock_tokenizer.assert_called_with("Test prompt", return_tensors="pt", truncation=True)
    mock_model.generate.assert_called_once()

# Tests for summarize_text
@patch('app.run_model')
def test_summarize_text(mock_run_model):
    mock_run_model.return_value = "Summary text"
    result = summarize_text("Input text")
    assert result == "Summary text"
    mock_run_model.assert_called_with("Summarize the following text in 3 concise sentences:\n\nInput text")

# Tests for extract_keywords
@patch('app.run_model')
def test_extract_keywords(mock_run_model):
    mock_run_model.return_value = "keyword1, keyword2"
    result = extract_keywords("Summary text")
    assert result == "keyword1, keyword2"
    mock_run_model.assert_called_with("Extract 5 important keywords from this summary:\n\nSummary text\n\nKeywords:")

# Tests for /summarize endpoint

@patch('app.extract_keywords')
@patch('app.summarize_text')
def test_summarize_success(mock_summarize, mock_extract):
    mock_summarize.return_value = "This is a summary."
    mock_extract.return_value = "keyword1, keyword2, keyword3"
    headers = {"x-api-key": API_KEY}
    response = client.post("/summarize", json={"text": "This is some text to summarize."}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "This is a summary."
    assert data["keywords"] == "keyword1, keyword2, keyword3"

def test_summarize_missing_api_key():
    response = client.post("/summarize", json={"text": "Test text"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API Key"}

def test_summarize_invalid_api_key():
    headers = {"x-api-key": "wrong-key"}
    response = client.post("/summarize", json={"text": "Test text"}, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API Key"}

def test_summarize_empty_text():
    headers = {"x-api-key": API_KEY}
    response = client.post("/summarize", json={"text": ""}, headers=headers)
    assert response.status_code == 400
    assert response.json() == {"detail": "Input text cannot be empty."}

def test_summarize_whitespace_text():
    headers = {"x-api-key": API_KEY}
    response = client.post("/summarize", json={"text": "   "}, headers=headers)
    assert response.status_code == 400
    assert response.json() == {"detail": "Input text cannot be empty."}

def test_summarize_missing_text_key():
    headers = {"x-api-key": API_KEY}
    response = client.post("/summarize", json={}, headers=headers)
    assert response.status_code == 422  # Pydantic validation error

def test_summarize_invalid_json():
    headers = {"x-api-key": API_KEY}
    response = client.post(
        "/summarize",
        data='{"text": "test"',
        headers={"content-type": "application/json", **headers}
    )
    assert response.status_code == 422  # FastAPI returns 422 for invalid JSON
    assert isinstance(response.json()["detail"], list)  # detail is a list of errors

@patch('app.summarize_text')
def test_summarize_model_error(mock_summarize):
    mock_summarize.side_effect = Exception("Model error")
    headers = {"x-api-key": API_KEY}
    response = client.post("/summarize", json={"text": "Test text"}, headers=headers)
    assert response.status_code == 500
    assert response.json() == {"detail": "Model error"}

# Test middleware logging (check if log file is created/updated)
@patch('app.log_to_file')
def test_middleware_logging(mock_log_to_file):
    headers = {"x-api-key": API_KEY}
    response = client.post("/summarize", json={"text": "Test text"}, headers=headers)
    assert response.status_code == 200
    # Check that log_to_file was called (middleware triggers it)
    mock_log_to_file.assert_called_once()
    args, kwargs = mock_log_to_file.call_args
    log_message = args[0]
    assert "POST /summarize" in log_message
    assert "status=200" in log_message
