from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from database import SessionLocal, Book as BookModel, Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI()

class Book(BaseModel):
    title: str
    author: str
    year: int
    
# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# create book route
@app.post("/books")
async def create_book_api(book: Book, db: Session = Depends(get_db)):
    try:
        db_book = BookModel(title=book.title, author=book.author, year=book.year)
        db.add(db_book)
        db.commit()
        db.refresh(db_book)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Book with this title already exists")
    return {"message": "Book added successfully", "task": {"title": db_book.title}}

# Get all books
@app.get("/books")
async def get_all_books(db: Session = Depends(get_db)):
    books = db.query(BookModel).all()
    return {"books": [{"id": b.id, "title": b.title, "author": b.author, "year": b.year} for b in books]}

# Delete books by it
@app.delete("/books/{id}")
async def delete_book(id: int, db: Session = Depends(get_db)):
    existing_book = db.query(BookModel).filter(BookModel.id == id).first()
    if not existing_book:
        raise HTTPException(status_code=404, detail="Book not found")

    db.delete(existing_book)
    db.commit()

    return {"message": f"Book with id {id} deleted successfully"}