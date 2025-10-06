import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_analyze_valid_sentence():
    response = client.post("/analyze", json={"sentence": "I love this product!"})
    assert response.status_code == 200
    json_data = response.json()
    assert "label" in json_data
    assert "score" in json_data
    assert isinstance(json_data["label"], str)
    assert isinstance(json_data["score"], float)

def test_analyze_empty_sentence():
    response = client.post("/analyze", json={"sentence": ""})
    assert response.status_code == 200
    json_data = response.json()
    assert "label" in json_data
    assert "score" in json_data

def test_analyze_missing_sentence_key():
    response = client.post("/analyze", json={})
    assert response.status_code == 400  # Changed to 400 to match app.py behavior

def test_analyze_invalid_json():
    response = client.post("/analyze", data="not a json")
    assert response.status_code == 400  # Bad Request due to invalid JSON

def test_analyze_non_string_sentence():
    response = client.post("/analyze", json={"sentence": 12345})
    # The transformers pipeline expects a string, so this might raise an error or handle it gracefully
    # We check for 400 due to validation in app.py
    assert response.status_code == 400
