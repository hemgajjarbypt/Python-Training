from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel
import logging
import asyncio
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

API_KEY = "my-secret-key"
    
async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

# Load model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="api_logs.log",
    filemode="a",
)

async def log_to_file(message: str):
    """Writes logs asynchronously using asyncio.to_thread"""
    await asyncio.to_thread(logging.info, message)

# Initialize FastAPI app
app = FastAPI(title="Text Summarization & Keyword Extraction API")

# Request body schema
class TextRequest(BaseModel):
    text: str
    
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

def run_model(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def summarize_text(text: str) -> str:
    prompt = f"Summarize the following text in 3 concise sentences:\n\n{text}"
    return run_model(prompt)

def extract_keywords(summary: str) -> str:
    prompt = f"Extract 5 important keywords from this summary:\n\n{summary}\n\nKeywords:"
    return run_model(prompt)

@app.post("/summarize", dependencies=[Depends(verify_api_key)])
async def summarize_and_extract(req: TextRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        summary = summarize_text(text)
        keywords = extract_keywords(summary)
        return {
            "summary": summary,
            "keywords": keywords
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
