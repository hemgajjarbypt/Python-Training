import os
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

TEST_USERS_FILE = os.path.join(os.getcwd(), "users.json")

@pytest.fixture(autouse=True)
def run_around_tests():
    # Setup: remove users.json before each test
    if os.path.exists(TEST_USERS_FILE):
        os.remove(TEST_USERS_FILE)
    yield
    # Teardown: remove users.json after each test
    if os.path.exists(TEST_USERS_FILE):
        os.remove(TEST_USERS_FILE)

def test_create_user():
    response = client.post("/user/", json={"name": "Alice", "age": 30})
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "User added successfully"
    assert data["user"]["name"] == "Alice"
    assert data["user"]["age"] == 30

def test_get_user_success():
    client.post("/user/", json={"name": "Bob", "age": 25})
    response = client.get("/user/Bob")
    assert response.status_code == 200
    data = response.json()
    assert data["user"]["name"] == "Bob"
    assert data["user"]["age"] == 25

def test_get_user_not_found():
    response = client.get("/user/Charlie")
    assert response.status_code == 404
    assert response.json()["detail"] == "No users found"

    # Add a user, then search for a non-existent user
    client.post("/user/", json={"name": "Dana", "age": 40})
    response = client.get("/user/NonExistent")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

def test_create_multiple_users():
    users = [
        {"name": "Eve", "age": 22},
        {"name": "Frank", "age": 33},
        {"name": "Grace", "age": 44}
    ]
    for user in users:
        resp = client.post("/user/", json=user)
        assert resp.status_code == 200
    # Check all users exist
    for user in users:
        resp = client.get(f"/user/{user['name']}")
        assert resp.status_code == 200
        assert resp.json()["user"]["age"] == user["age"]

def test_create_user_invalid_payload():
    # Missing age
    resp = client.post("/user/", json={"name": "Henry"})
    assert resp.status_code == 422
    # Missing name
    resp = client.post("/user/", json={"age": 50})
    assert resp.status_code == 422
    # Wrong type
    resp = client.post("/user/", json={"name": "Ivy", "age": "notanint"})
    assert resp.status_code == 422

def test_users_json_corrupted():
    # Write corrupted JSON
    with open(TEST_USERS_FILE, "w") as f:
        f.write("not a json")
    # Should handle gracefully and allow adding user
    resp = client.post("/user/", json={"name": "Jack", "age": 60})
    assert resp.status_code == 200
    # Should handle gracefully and allow getting user
    resp = client.get("/user/Jack")
    assert resp.status_code == 200
    assert resp.json()["user"]["name"] == "Jack"
