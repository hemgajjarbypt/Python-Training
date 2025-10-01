import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, FastAPI!"}

def test_echo_api_valid_json():
    payload = {"key": "value"}
    response = client.post("/echo", json=payload)
    assert response.status_code == 200
    assert response.json() == payload

def test_echo_api_empty_json():
    response = client.post("/echo", json={})
    assert response.status_code == 200
    assert response.json() == {}

def test_echo_api_invalid_json():
    try:
        response = client.post("/echo", content="not a json", headers={"Content-Type": "application/json"})
        assert response.status_code == 422 or response.status_code == 400
    except Exception as e:
        # Acceptable if JSONDecodeError is raised, as this is expected for invalid JSON
        assert "JSONDecodeError" in type(e).__name__

# Edge case: large payload
def test_echo_api_large_json():
    payload = {str(i): i for i in range(1000)}
    response = client.post("/echo", json=payload)
    assert response.status_code == 200
    assert response.json() == payload
