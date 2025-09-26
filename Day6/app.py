from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal, Task as TaskModel, Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI()

class Task(BaseModel):
    title: str
    
# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# create task route
@app.post("/task/")
async def create_task_api(task: Task, db: Session = Depends(get_db)):
    
    existing_task = db.query(TaskModel).filter(TaskModel.title == task.title).first()
    if existing_task:
        raise HTTPException(status_code=400, detail="Task already exists")

    db_task = TaskModel(title=task.title)
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    
    return {"message": "Task added successfully", "task": {"title": db_task.title}}

# Get all tasks 
@app.get("/tasks")
async def get_all_tasks(db: Session = Depends(get_db)):
    tasks = db.query(TaskModel).all()
    return {"tasks": [{"id": t.id, "title": t.title} for t in tasks]}
