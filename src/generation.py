"""
This module generates responses from user queries based on the retrieved context.
"""
import os
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def generate_response(prompt: str, context: str) -> str:
    """
    Generates a response using the Gemini LLM based on the provided context.

    Args:
        prompt (str): The question to ask the LLM.
        context (str): The context to use for answering the question.

    Returns:
        str: The response from the LLM.
    """
    # Get API key and initialize client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not found. Please set it in your .env file.")

    # Initialize the Gemini client    
    client = genai.Client(api_key=api_key)
    
    # Construct the prompt for the LLM
    final_prompt = f"""You are a helpful assistant analyzing financial documents.
        Answer the user's question using only the provided context. 
        If the answer is not in the context, say "I cannot answer this based on the provided documents."
        Context:
        {context}
        Question:
        {prompt}
    """
    # Call the Gemini API
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=final_prompt
    )
    
    # Return the response text
    return response.text
