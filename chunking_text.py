import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def process_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    cleaned_text = clean_text(raw_text)
    text_chunks = chunk_text(cleaned_text)
    
    print(f"Total number of chunks: {len(text_chunks)}")
    
    return text_chunks
