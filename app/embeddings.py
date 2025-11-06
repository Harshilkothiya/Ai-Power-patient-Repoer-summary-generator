import os
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text: str):
    """
    Generate a text embedding using Google's Gemini model.
    Input: text (string)
    Output: list[float] — embedding vector
    """
    try:
        # Use Gemini text embedding model
        model = "models/text-embedding-004"
        response = genai.embed_content(
            model=model,
            content=text
        )
        return response['embedding']
    except Exception as e:
        raise RuntimeError(f"Gemini embedding failed: {str(e)}")
