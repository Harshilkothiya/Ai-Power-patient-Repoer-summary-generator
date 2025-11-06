import os
import google.generativeai as genai
from typing import List, Optional
from langchain.schema import Document

class LLMClient:
    def __init__(self):
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY environment variable")
        genai.configure(api_key=api_key)
        
        # Choose model (allow override via GEMINI_MODEL env var)
        model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro")
        try:
            self.model = genai.GenerativeModel(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model '{model_name}': {e}")

    def _format_documents(self, docs: List[Document]) -> str:
        """Format documents into a string for the prompt"""
        formatted_docs = []
        for doc in docs:
            metadata = doc.metadata
            formatted_docs.append(
                f"{metadata['doc_type'].capitalize()} ({metadata['date']}): {doc.page_content}"
            )
        return "\n\n".join(formatted_docs)

    def generate_daily_summary(self, docs: List[Document], date: Optional[str] = None) -> str:
        """Generate a daily summary from documents"""
        prompt = f"""You are a medical summarization assistant. 
        Based on the following patient records, create a clear and concise daily summary.
        Focus on key medical events, treatments, and patient status for {date or 'the specified period'}.

        Patient Records:
        {self._format_documents(docs)}

        Create a structured summary that includes:
        1. Patient Status
        2. Medications & Treatments
        3. Test Results (if any)
        4. Key Observations
        """
        
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                candidate_count=1,
            )
        )
        return response.text

    def generate_discharge_summary(self, docs: List[Document]) -> str:
        """Generate a comprehensive discharge summary"""
        prompt = f"""You are a medical summarization assistant.
        Based on the following patient records, create a comprehensive discharge summary
        that provides a complete overview of the patient's hospitalization journey.

        Patient Records:
        {self._format_documents(docs)}

        Create a structured summary that includes:
        1. Admission Details
        2. Key Diagnoses
        3. Treatment Timeline
        4. Procedures Performed
        5. Medications
        6. Patient Progress
        7. Discharge Recommendations
        """
        
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                candidate_count=1,
            )
        )
        return response.text
