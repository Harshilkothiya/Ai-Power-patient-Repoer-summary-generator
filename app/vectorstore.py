import chromadb
from chromadb.utils import embedding_functions
from embeddings import get_embedding
import os

# Create or load a persistent Chroma database
CHROMA_DIR = os.path.join(os.getcwd(), "data", "chroma_store")
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Create (or reuse) a collection
collection = client.get_or_create_collection(name="patient_documents")

def add_documents(documents, metadatas=None):
    """
    Add documents to the vector store.
    Args:
        documents (list[str]): List of text chunks to store
        metadatas (list[dict]): Optional list of metadata dicts
    """
    if not documents:
        raise ValueError("No documents provided for storage.")

    embeddings = [get_embedding(doc) for doc in documents]
    ids = [f"doc_{i}" for i in range(len(documents))]
    collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print(f"✅ Added {len(documents)} documents to vector store.")

def retrieve_context(query, top_k=3):
    """
    Retrieve top-k most relevant documents for a given query.
    Args:
        query (str): The search query
        top_k (int): Number of documents to retrieve
    Returns:
        list[str]: List of relevant document texts
    """
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    docs = results.get("documents", [[]])[0]
    return docs
