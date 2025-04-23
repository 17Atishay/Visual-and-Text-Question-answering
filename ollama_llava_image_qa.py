import requests
import base64
from typing import List

def query_llava_ollama(image_bytes: bytes, question: str, ollama_url: str = 'http://localhost:11434/api/generate', model: str = 'llava') -> str:
    """
    Sends an image and question to the local Ollama server running the Llava model and returns the answer.
    """
    # Encode image as base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        'model': model,
        'prompt': question,
        'images': [image_b64],
        'stream': False
    }
    response = requests.post(ollama_url, json=payload)
    response.raise_for_status()
    return response.json().get('response', '')
