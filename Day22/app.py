from fastapi import FastAPI, Request, HTTPException
import logging
import asyncio
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="api_logs.log",
    filemode="a",
)

app = FastAPI()

async def log_to_file(message: str):
    """Writes logs asynchronously using asyncio.to_thread"""
    await asyncio.to_thread(logging.info, message)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()

    response = await call_next(request)

    duration = (datetime.utcnow() - start_time).total_seconds()

    log_message = (
        f"{request.client.host} - {request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.3f}s"
    )

    asyncio.create_task(log_to_file(log_message))

    return response

@app.get("/")
async def hello_route():
    try:
        return {"msg": "Hello World"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))