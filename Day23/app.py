from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="LLM Question Answering API")

API_KEY = "my-secret-key"

class QuestionRequest(BaseModel):
    question: str
    
async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

qa_pipeline = pipeline("text-generation", model="bigscience/bloom-560m")

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