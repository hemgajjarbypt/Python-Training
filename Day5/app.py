import json
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    age: int

# create user route
@app.post("/user/")
async def create_user_api(user: User):
    
    file_path = os.path.join(os.getcwd(), "users.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                users = json.load(file)
            except json.JSONDecodeError:
                users = []
    else:
        users = []
        
    users.append(user.model_dump())
    
    with open(file_path, "w") as file:
        json.dump(users, file, indent=4)

    return {"message": "User added successfully", "user": user}

# Get user by user name
@app.get("/user/{name}")
async def get_user_api(name: str):
    file_path = os.path.join(os.getcwd(), "users.json")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="No users found")
    
    with open(file_path, "r") as file:
        try:
            users = json.load(file)
        except json.JSONDecodeError:
            users = []
            
    for user in users:
        if user.get("name") == name:
            return {"user": user}
    
    raise HTTPException(status_code=404, detail=f"User with name '{name}' not found")
