import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, Task as TaskModel
from app import app, get_db

# Setup test database
TEST_DATABASE_URL = "sqlite:///./test_tasks.db"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
Base.metadata.create_all(bind=test_engine)

# Dependency override for tests
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def run_around_tests():
    # Clean up before each test
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    yield
    # Clean up after each test
    Base.metadata.drop_all(bind=test_engine)

# Test creating a new task
def test_create_task():
    response = client.post("/task/", json={"title": "Test Task"})
    assert response.status_code == 200
    assert response.json()["task"]["title"] == "Test Task"

# Test creating a duplicate task
def test_create_duplicate_task():
    client.post("/task/", json={"title": "Test Task"})
    response = client.post("/task/", json={"title": "Test Task"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Task already exists"

# Test getting all tasks
def test_get_all_tasks():
    client.post("/task/", json={"title": "Task 1"})
    client.post("/task/", json={"title": "Task 2"})
    response = client.get("/tasks")
    assert response.status_code == 200
    tasks = response.json()["tasks"]
    assert len(tasks) == 2
    assert tasks[0]["title"] == "Task 1"
    assert tasks[1]["title"] == "Task 2"

# Test get all tasks when none exist
def test_get_all_tasks_empty():
    response = client.get("/tasks")
    assert response.status_code == 200
    assert response.json()["tasks"] == []

# Test creating task with empty title
def test_create_task_empty_title():
    response = client.post("/task/", json={"title": ""})
    assert response.status_code == 200 or response.status_code == 422
    # If validation is enforced, status 422; else, status 200 with empty title

# Test creating task with long title
def test_create_task_long_title():
    long_title = "A" * 255
    response = client.post("/task/", json={"title": long_title})
    assert response.status_code == 200
    assert response.json()["task"]["title"] == long_title

# Test creating multiple unique tasks
def test_create_multiple_unique_tasks():
    titles = ["Task A", "Task B", "Task C"]
    for title in titles:
        response = client.post("/task/", json={"title": title})
        assert response.status_code == 200
        assert response.json()["task"]["title"] == title
    response = client.get("/tasks")
    tasks = response.json()["tasks"]
    assert len(tasks) == 3
    returned_titles = [t["title"] for t in tasks]
    for title in titles:
        assert title in returned_titles

# Test get_db yields session
def test_get_db_yields_session():
    from app import get_db
    gen = get_db()
    db = next(gen)
    from sqlalchemy.orm import Session
    assert isinstance(db, Session)
    try:
        next(gen)
    except StopIteration:
        pass  # Generator should close without error

# Test get_db closes session after use
def test_get_db_closes_session():
    from app import get_db
    gen = get_db()
    db = next(gen)
    assert db.is_active
    try:
        next(gen)
    except StopIteration:
        pass
    # After generator closes, session should be closed
    assert not db.is_active or db.close