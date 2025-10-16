from fastapi import FastAPI, Request, HTTPException, Depends, Header
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI(title="LLM Question Answering API")

API_KEY = "my-secret-key"

class QuestionRequest(BaseModel):
    question: str
    
async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

qa_pipeline = pipeline("text-generation", model="bigscience/bloom-560m")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.post("/sentiment")
async def analyze_api(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if 'sentence' not in body:
        raise HTTPException(status_code=400, detail="Missing 'sentence' key")
    sentence = body['sentence']
    if not isinstance(sentence, str):
        raise HTTPException(status_code=400, detail="'sentence' must be a string")
    try:
        result = sentiment_pipeline(sentence)
    except Exception:
        raise HTTPException(status_code=500, detail="Error processing sentence")
    return result[0]

@app.post("/qa", dependencies=[Depends(verify_api_key)])
async def ask_question(request: QuestionRequest):
    question = request.question
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        response = qa_pipeline(question, max_new_tokens=100)
        answer = response[0]["generated_text"]
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))