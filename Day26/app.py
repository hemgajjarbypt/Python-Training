from fastapi import FastAPI, Request, HTTPException
import logging
import asyncio
from datetime import datetime
from starlette.responses import Response

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

    try:
        response: Response = await call_next(request)
    except Exception as e:
        # Handle unhandled exceptions (500)
        response = Response(content=str(e), status_code=500)
        # Log the error
        error_message = (
            f"{request.client.host} - {request.method} {request.url.path} "
            f"status=500 exception={str(e)}"
        )
        asyncio.create_task(log_to_file(error_message))
        raise e

    duration = (datetime.utcnow() - start_time).total_seconds()

    log_message = (
        f"{request.client.host} - {request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.3f}s"
    )

    # Log all invalid or error responses (status >= 400)
    if response.status_code >= 400:
        asyncio.create_task(log_to_file("INVALID: " + log_message))
    else:
        asyncio.create_task(log_to_file(log_message))

    return response

@app.get("/")
async def hello_route():
    return {"msg": "Hello World"}

@app.get("/error")
async def error_route():
    # Example endpoint to test 400 or 500 errors
    raise HTTPException(status_code=400, detail="This is a test bad request")