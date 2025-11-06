import json

# Function to clean and format model responses
def format_response(response):
    """
    Format the model's raw output into clean, readable text.
    Handles JSON and plain text responses.
    """
    if not response:
        return "No response generated."

    # Try to parse JSON if model response is structured
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "output" in data:
            return data["output"]
        elif isinstance(data, list):
            return "\n".join(str(item) for item in data)
        else:
            return json.dumps(data, indent=2)
    except Exception:
        pass  # Continue to plain text

    # Clean up raw text (remove special tokens or unwanted parts)
    clean_text = (
        response.replace("```", "")
        .replace("json", "")
        .replace("Output:", "")
        .replace("Response:", "")
        .strip()
    )

    # Capitalize first letter and ensure proper formatting
    if clean_text and not clean_text[0].isupper():
        clean_text = clean_text[0].upper() + clean_text[1:]

    return clean_text


# Function to pretty print retrieved context
def display_context(contexts):
    """
    Display retrieved context snippets from vectorstore neatly.
    """
    if not contexts:
        return "No context found."

    formatted_context = "\n\n".join(
        [f"🔹 **Document {i+1}:**\n{ctx}" for i, ctx in enumerate(contexts)]
    )
    return formatted_context


# Function to merge prompt and context for RAG input
def combine_prompt_with_context(query, context):
    """
    Combine user query with retrieved knowledge for final model input.
    """
    return f"""
You are an intelligent assistant using retrieved knowledge to answer accurately.

Context:
{context}

Question:
{query}

Answer in a clear, concise, and factual manner.
""".strip()
