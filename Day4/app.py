from fastapi import FastAPI, Request

app = FastAPI()

# basic Hello route
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

# echo route
@app.post("/echo")
async def echo_api(request: Request):
    body = await request.json()
    return body