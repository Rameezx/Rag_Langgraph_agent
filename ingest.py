"""
Step 3: Standalone RAG pipeline (no LLM yet).
This script:
  1. Loads your PDFs/notes from the data/ folder
  2. Splits them into small chunks
  3. Turns each chunk into numbers (embeddings)
  4. Stores those numbers in a local vector database (Chroma)
  5. Lets you search ("retrieve") the most relevant chunks for a question

Run it directly to test: python ingest.py
"""

import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data"          # put your PDFs here
CHROMA_DIR = "chroma_db"   # Chroma will save its files here


def load_documents():
    """Step 1: Loader — reads every PDF in data/ and turns it into plain text."""
    if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR):
        raise FileNotFoundError(
            f"No files found in '{DATA_DIR}/'. Create the folder and put at least one PDF in it."
        )

    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s) from '{DATA_DIR}/'")
    return documents


def split_documents(documents):
    """Step 2: Splitter — cuts the text into small overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # characters per chunk
        chunk_overlap=50,    # slight overlap so context isn't cut mid-sentence
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)")
    return chunks


def build_vector_store(chunks):
    """Steps 3 & 4: Embedder + Vector store — turns chunks into numbers and saves them."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"Saved vector store to '{CHROMA_DIR}/'")
    return vector_store


def load_existing_vector_store():
    """Reuse a vector store you already built, instead of re-processing PDFs every time."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def retrieve(vector_store, query, k=3):
    """Step 5: Retriever — given a question, returns the k most relevant chunks."""
    results = vector_store.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    # Build (or rebuild) the vector store from your PDFs
    docs = load_documents()
    chunks = split_documents(docs)
    store = build_vector_store(chunks)

    # Test retrieval alone — no LLM involved yet
    test_question = "What certification the person have?"   # <-- change this to a question about YOUR notes
    print(f"\nQuery: {test_question}\n")

    results = retrieve(store, test_question, k=3)
    for i, chunk in enumerate(results, start=1):
        print(f"--- Result {i} (source: {chunk.metadata.get('source', 'unknown')}) ---")
        print(chunk.page_content)
        print()