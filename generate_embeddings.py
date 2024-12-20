from sentence_transformers import SentenceTransformer
from chunking_text import process_text

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(file_path):
    text_chunks = process_text(file_path)
    embeddings = model.encode(text_chunks, show_progress_bar=True)    
    return embeddings

if __name__ == "__main__":
    file_path = "extracted_annual_reports.txt"
    embeddings = generate_embeddings(file_path)
