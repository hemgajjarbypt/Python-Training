import pytest
from fastapi.testclient import TestClient
from app import app
from database import Base, engine, SessionLocal
from sqlalchemy.orm import sessionmaker

client = TestClient(app)

# Setup and teardown for test DB
@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Recreate tables before each test
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    # Drop tables after each test
    Base.metadata.drop_all(bind=engine)

# Helper to add a book directly
@pytest.fixture
def add_book():
    def _add_book(title, author, year):
        db = SessionLocal()
        db_book = db.query(Base.classes.books).filter_by(title=title).first()
        if not db_book:
            db.execute(f"INSERT INTO books (title, author, year) VALUES ('{title}', '{author}', {year})")
            db.commit()
        db.close()
    return _add_book

# Test book creation
def test_create_book():
    response = client.post("/books", json={"title": "Book1", "author": "Author1", "year": 2020})
    assert response.status_code == 200
    assert response.json()["message"] == "Book added successfully"
    assert response.json()["task"]["title"] == "Book1"

# Test duplicate book creation
def test_create_duplicate_book():
    client.post("/books", json={"title": "Book1", "author": "Author1", "year": 2020})
    response = client.post("/books", json={"title": "Book1", "author": "Author2", "year": 2021})
    assert response.status_code == 500 or response.status_code == 400

# Test get all books
def test_get_all_books():
    client.post("/books", json={"title": "Book1", "author": "Author1", "year": 2020})
    client.post("/books", json={"title": "Book2", "author": "Author2", "year": 2021})
    response = client.get("/books")
    assert response.status_code == 200
    books = response.json()["books"]
    assert len(books) == 2
    assert books[0]["title"] == "Book1"
    assert books[1]["title"] == "Book2"

# Test delete book
def test_delete_book():
    post_resp = client.post("/books", json={"title": "Book1", "author": "Author1", "year": 2020})
    get_resp = client.get("/books")
    book_id = get_resp.json()["books"][0]["id"]
    del_resp = client.delete(f"/books/{book_id}")
    assert del_resp.status_code == 200
    assert "deleted successfully" in del_resp.json()["message"]
    # Confirm deletion
    get_resp2 = client.get("/books")
    assert len(get_resp2.json()["books"]) == 0

# Test delete non-existent book
def test_delete_nonexistent_book():
    response = client.delete("/books/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Book not found"

# Test invalid payloads
@pytest.mark.parametrize("payload", [
    {"title": "Book", "author": "Author", "year": "not_a_year"},
    {"title": "Book"},
])
def test_create_book_invalid_payload(payload):
    response = client.post("/books", json=payload)
    assert response.status_code == 422
