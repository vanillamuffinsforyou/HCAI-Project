import os
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text_into_chunks(text, chunk_size=500):
    """
    Split the text into chunks of approximately chunk_size characters.
    This simple approach splits on periods. You can adjust logic as needed.
    """
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def create_vector_store(chunks, model_name='all-MiniLM-L6-v2'):
    """Create embeddings for each chunk and build a FAISS index."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return embeddings, index

def main():
    pdf_path = "sorting_algorithms.pdf"  # Make sure the PDF is in your project folder
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    print("Splitting text into chunks...")
    chunks = split_text_into_chunks(text)
    print(f"Created {len(chunks)} chunks.")
    print("Generating embeddings and creating vector store...")
    embeddings, index = create_vector_store(chunks)
    
    # Save chunks and embeddings for later use
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    np.save("embeddings.npy", embeddings)
    # Save the FAISS index
    faiss.write_index(index, "faiss.index")
    print("Preprocessing complete. Files saved: chunks.pkl, embeddings.npy, and faiss.index")

if __name__ == "__main__":
    main()
