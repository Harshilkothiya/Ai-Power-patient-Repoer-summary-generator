# app/ingest.py
import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict
from app.utils import chunk_text

def ingest_csv(filepath: str, out_jsonl: str, doc_type: str):
    df = pd.read_csv(filepath)
    items = []
    for _, row in df.iterrows():
        patient_id = str(row.get('patient_id', 'unknown'))
        text = str(row.get('text', ''))
        date = row.get('date', '')
        if not text.strip():
            continue
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        for i, chunk in enumerate(chunks):
            items.append({
                "patient_id": patient_id,
                "doc_type": doc_type,
                "date": str(date),
                "chunk_id": f"{os.path.basename(filepath)}_{_}_{i}",
                "text": chunk
            })
    with open(out_jsonl, 'a', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # example usage
    ingest_csv("data/doctor_notes.csv", "data/ingested.jsonl", doc_type="doctor_note")
    ingest_csv("data/prescriptions.csv", "data/ingested.jsonl", doc_type="prescription")
    ingest_csv("data/lab_reports.csv", "data/ingested.jsonl", doc_type="lab_report")
