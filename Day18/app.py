import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_text_files(folder_path):
    texts = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                texts.append(f.read())
                filenames.append(filename)
    return filenames, texts

def main():    
    folder_path = "Data"
    filenames, texts = load_text_files(folder_path)

    embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    embeddings_np = np.array(embeddings).astype("float32")

    index.add(embeddings_np)
    print(f"Stored {index.ntotal} documents in FAISS index.")

    query = "Which files talk about AI?"
    query_vector = model.encode([query])

    k = 3  # top 3 results
    distances, indices = index.search(np.array(query_vector).astype("float32"), k)

    print("\nTop matching files:\n")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {filenames[idx]}  (distance={distances[0][i]:.4f})")
        
if __name__ == '__main__':
    main()