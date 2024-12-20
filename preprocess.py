import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

with open("extracted_annual_reports.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

cleaned_text = clean_text(raw_text)
text_chunks = chunk_text(cleaned_text)

print(f"Total chunks created: {len(text_chunks)}")


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(text_chunks, show_progress_bar=True)
print(f"Generated embeddings for {len(embeddings)} text chunks.")



embeddings_np = np.array(embeddings).astype('float32')

index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)

print("Embeddings stored in FAISS.")


def retrieve_relevant_chunks(query, model, index, chunks, k=3):
    """Retrieve top-k relevant text chunks for a query."""
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

query = "What is the company's net profit for the year?"
relevant_chunks = retrieve_relevant_chunks(query, model, index, text_chunks)

print("Top Relevant Chunks:")
for i, chunk in enumerate(relevant_chunks, 1):
    print(f"\n--- Chunk {i} ---\n{chunk}")
