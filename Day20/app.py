from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader

# --- Step 1: Extract Text from PDF ---
def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# --- Step 2: Split into Chunks ---
def chunk_text(text, chunk_size=400):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# --- Step 3: Create Embeddings and FAISS Index ---
def create_faiss_index(chunks):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embedder


# --- Step 4: Retrieve Relevant Chunks ---
def retrieve_relevant_chunks(query, model, index, chunks, top_k=3):
    query_emb = model.encode([query])
    distances, indices = index.search(np.array(query_emb), top_k)
    return [chunks[i] for i in indices[0]]


# --- Step 5: Use RAG Model (Roberta-based QA) ---
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result["answer"]


# --- Step 6: Combine into One Function ---
def rag_qa_from_pdf(pdf_path, question, top_k=3):
    print("📘 Extracting text...")
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    print("🔍 Creating embeddings...")
    index, emb_model = create_faiss_index(chunks)

    print("🎯 Retrieving relevant context...")
    relevant_chunks = retrieve_relevant_chunks(question, emb_model, index, chunks, top_k)
    combined_context = " ".join(relevant_chunks)

    print("🧠 Generating answer using RAG model...")
    answer = answer_question(question, combined_context)
    return answer


def main():
    pdf_path = "sample.pdf"
    question = "What architecture does this paper introduce?"
    answer = rag_qa_from_pdf(pdf_path, question)
    print("\nAnswer:", answer)
    
if __name__ == '__main__':
    main()