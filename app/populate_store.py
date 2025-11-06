# app/populate_store.py
import json
import numpy as np
from app.embeddings import Embedder
from app.vectorstore import FaissStore

def populate(jsonl_path="data/ingested.jsonl"):
    docs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            docs.append(json.loads(line))
    texts = [d['text'] for d in docs]
    embedder = Embedder()
    embs = embedder.embed_texts(texts)
    dim = embs.shape[1]
    store = FaissStore(dim)
    metadata = [{"patient_id": d["patient_id"], "doc_type": d["doc_type"], "date": d["date"], "chunk_id": d["chunk_id"], "text": d["text"]} for d in docs]
    store.add(embs, metadata)

if __name__ == "__main__":
    populate()
    