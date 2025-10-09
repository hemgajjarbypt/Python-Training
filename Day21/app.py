from fastapi import FastAPI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from pydantic import BaseModel
import os

app = FastAPI(title="PDF QA Bot (LangChain + Hugging Face)")

PDF_PATH = "sample.pdf"
VECTOR_STORE_PATH = "faiss_store"
TOP_K = 3

# --- Initialize global variables ---
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

vectorstore = None

class QuestionRequest(BaseModel):
    question: str
    
def clean_text(text):
    import re
    # remove unwanted tokens
    text = re.sub(r"<pad>|<EOS>", " ", text)
    # remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --- Load and Embed PDF ---
def initialize_vectorstore():
    global vectorstore

    if os.path.exists(VECTOR_STORE_PATH):
        print("✅ Loading existing FAISS index...")
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("📘 Loading and processing PDF...")
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        print("✂️ Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        for c in chunks:
            c.page_content = clean_text(c.page_content)

        print("🔍 Creating embeddings and saving FAISS index...")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local(VECTOR_STORE_PATH)
        print("✅ FAISS index saved!")


@app.on_event("startup")
def on_startup():
    initialize_vectorstore()


# --- /ask endpoint ---
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        if not vectorstore:
            return {"error": "Vector store not initialized."}

        # Retrieve top-k chunks
        docs = vectorstore.similarity_search(request.question, k=TOP_K)
        context = " ".join([d.page_content for d in docs])

        # Use Hugging Face QA model
        result = qa_model(question=request.question, context=context)
        return {
            "question": request.question,
            "answer": result["answer"],
            "context_snippet": context[:400] + "..."
        }
    except Exception as e:
        return {"error": str(e)}
