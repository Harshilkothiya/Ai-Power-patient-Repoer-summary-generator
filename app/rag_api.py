from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

# Load .env into environment variables early so modules that read os.getenv() get values
load_dotenv()

from data_processor import MedicalDataProcessor
from vector_store import VectorStoreManager
from llm_client import LLMClient

app = Flask(__name__)

# Initialize components
data_processor = MedicalDataProcessor()
vector_store = VectorStoreManager(
    persist_dir=os.path.join(os.path.dirname(__file__), "data", "chroma_store")
)
llm_client = LLMClient()

# Load initial data
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "patient_records.json")
if os.path.exists(DATA_PATH):
    records = data_processor.load_patient_records(DATA_PATH)
    documents = data_processor.process_patient_records(records)
    vector_store.add_documents(documents)

@app.route("/patient/<patient_id>/daily-summary", methods=["GET"])
def daily_summary(patient_id):
    date = request.args.get('date')
    
    # Retrieve relevant documents
    docs = vector_store.get_patient_documents(
        patient_id=patient_id,
        date=date,
        k=10
    )
    
    if not docs:
        return jsonify({
            "status": "error",
            "message": f"No records found for patient {patient_id}"
        }), 404
    
    # Generate summary
    summary = llm_client.generate_daily_summary(docs, date)
    
    return jsonify({
        "status": "success",
        "patient_id": patient_id,
        "date": date,
        "summary": summary
    })

@app.route("/patient/<patient_id>/discharge-summary", methods=["GET"])
def discharge_summary(patient_id):
    # Retrieve all patient documents
    docs = vector_store.get_patient_documents(
        patient_id=patient_id,
        k=20  # Get more documents for complete history
    )
    
    if not docs:
        return jsonify({
            "status": "error",
            "message": f"No records found for patient {patient_id}"
        }), 404
    
    # Generate summary
    summary = llm_client.generate_discharge_summary(docs)
    
    return jsonify({
        "status": "success",
        "patient_id": patient_id,
        "summary": summary
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Medical RAG Summarization API is running ✅",
        "endpoints": {
            "/patient/{id}/daily-summary": "GET - Get daily summary with optional date parameter",
            "/patient/{id}/discharge-summary": "GET - Get complete discharge summary"
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
