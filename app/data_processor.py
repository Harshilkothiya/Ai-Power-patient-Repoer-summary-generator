from typing import List, Dict, Any
from langchain.schema import Document
import json
import os

class MedicalDataProcessor:
        
    def load_patient_records(self, file_path: str) -> List[Dict[str, Any]]:
        """Load patient records from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading patient records: {e}")
            return []

    def process_document(self, text: str, metadata: Dict) -> List[Document]:
        """Create Document object from text"""
        return [Document(
            page_content=text,
            metadata={**metadata}
        )]

    def process_patient_records(self, records: List[Dict]) -> List[Document]:
        """Process all patient records into Document objects"""
        documents = []
        for patient in records:
            patient_id = patient["patient_id"]
            for doc in patient["documents"]:
                doc_with_metadata = self.process_document(
                    doc["text"],
                    {
                        "patient_id": patient_id,
                        "doc_type": doc["doc_type"],
                        "date": doc["date"]
                    }
                )
                documents.extend(doc_with_metadata)
        return documents