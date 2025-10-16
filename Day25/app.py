from fastapi import FastAPI, Request, HTTPException
from transformers import pipeline

app = FastAPI()

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