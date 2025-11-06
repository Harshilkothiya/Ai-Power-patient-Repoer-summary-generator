import json
import os
from typing import List, Optional, Dict
from langchain.schema import Document

class VectorStoreManager:
    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.documents: Dict[str, List[Document]] = {}
        self._load_or_create_vectorstore()

    def _load_or_create_vectorstore(self):
        """Load existing documents or create new store"""
        os.makedirs(self.persist_dir, exist_ok=True)
        store_file = os.path.join(self.persist_dir, "documents.json")
        
        if os.path.exists(store_file):
            with open(store_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for patient_id, docs in data.items():
                    self.documents[patient_id] = [
                        Document(
                            page_content=doc["content"],
                            metadata=doc["metadata"]
                        ) for doc in docs
                    ]

    def _save_store(self):
        """Save documents to persistent storage"""
        store_file = os.path.join(self.persist_dir, "documents.json")
        data = {}
        for patient_id, docs in self.documents.items():
            data[patient_id] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs
            ]
        
        with open(store_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_documents(self, documents: List[Document]):
        """Add documents to store"""
        for doc in documents:
            patient_id = doc.metadata.get("patient_id")
            if patient_id:
                if patient_id not in self.documents:
                    self.documents[patient_id] = []
                self.documents[patient_id].append(doc)
        self._save_store()

    def get_patient_documents(self, 
                            patient_id: str, 
                            doc_type: Optional[str] = None,
                            date: Optional[str] = None,
                            k: Optional[int] = None) -> List[Document]:
        """Get documents for specific patient with optional filters"""
        if patient_id not in self.documents:
            return []
            
        docs = self.documents[patient_id]
        
        # Apply filters
        if doc_type:
            docs = [d for d in docs if d.metadata.get("doc_type") == doc_type]
        if date:
            docs = [d for d in docs if d.metadata.get("date") == date]
            
        # Sort by date
        docs.sort(key=lambda x: x.metadata.get("date", ""))
        
        # Limit results if k is specified
        if k is not None:
            docs = docs[:k]
            
        return docs