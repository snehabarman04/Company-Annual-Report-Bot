import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from chunking_text import process_text
from generate_embeddings import generate_embeddings
from transformers import pipeline
import pickle

def load_embeddings(filename='embeddings.pkl'):
    with open(filename, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def create_faiss_index(embeddings):
    embeddings_np = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index

def retrieve_relevant_chunks(query, model, index, chunks, k=3):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def summarize_text(text, model='facebook/bart-large-cnn'):
    summarizer = pipeline('summarization', model=model)
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def generate_answer(relevant_chunks):
    combined_text = ' '.join(relevant_chunks)
    
    summary = summarize_text(combined_text)
    return summary

def main():
    file_path = "extracted_annual_reports.txt"

    text_chunks = process_text(file_path)
    embeddings = generate_embeddings(file_path)
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

    index = create_faiss_index(embeddings)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    query = "give me the net sales of the company in the entire year"

    relevant_chunks = retrieve_relevant_chunks(query, model, index, text_chunks)
    print(relevant_chunks)
    answer = generate_answer(relevant_chunks)

    print("answer: ", answer)
    

if __name__ == "__main__":
    main()
