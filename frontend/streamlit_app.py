import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

def display_summary(data):
    """Display the summary in a formatted way"""
    if not data:
        return
    
    st.subheader("Summary")
    summary = data.get("summary", "")
    
    # Split summary into sections and display
    sections = summary.split("\n\n")
    for section in sections:
        if section.strip():
            st.write(section)

st.set_page_config(
    page_title="Medical Scribe RAG Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical Scribe — RAG Assistant")
st.markdown("""
This AI assistant helps summarize patient medical records using advanced language models.
- Get daily summaries with optional date filtering
- Generate comprehensive discharge summaries
- Quick access to patient history
""")

with st.sidebar:
    st.header("Patient Information")
    patient_id = st.text_input("Patient ID")
    date = st.text_input("Date (YYYY-MM-DD) — optional", 
                        help="Filter daily summary by date")

col1, col2 = st.columns(2)

with col1:
    if st.button("🗓️ Get Daily Summary", use_container_width=True):
        if not patient_id:
            st.warning("⚠️ Please enter a patient ID")
        else:
            url = f"{API_BASE}/patient/{patient_id}/daily-summary"
            params = {}
            if date:
                params['date'] = date
            
            with st.spinner("🔄 Generating daily summary..."):
                try:
                    r = requests.get(url, params=params)
                    if r.status_code == 200:
                        data = r.json()
                        display_summary(data)
                    else:
                        st.error(f"❌ Error: {r.text}")
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")

with col2:
    if st.button("📋 Get Discharge Summary", use_container_width=True):
        if not patient_id:
            st.warning("⚠️ Please enter a patient ID")
        else:
            url = f"{API_BASE}/patient/{patient_id}/discharge-summary"
            
            with st.spinner("🔄 Generating discharge summary..."):
                try:
                    r = requests.get(url)
                    if r.status_code == 200:
                        data = r.json()
                        display_summary(data)
                    else:
                        st.error(f"❌ Error: {r.text}")
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit + LangChain + Mistral-7B 🚀")
