# Company Annual Report Bot

Here I modelled and made an LLM that is capable of answering all queries raised by the user from the company's annual report PDFs. 
The pipeline is:
It processes it into smaller, clean text chunks, generates embeddings for those chunks using a pre-trained SentenceTransformer model, and stores the embeddings in a FAISS index for efficient similarity search. 
Finally we are able to get a Question-Answering (Q&A) system where user queries are matched to relevant document chunks, and answers are generated based on the retrieved content.
