import os
import requests

OLLAMA_HOST = os.getenv(key="OLLAMA_HOST")

def search_internet(query=""):
    """Search internet
    
    Args:
        query (str): The search query to use
    """
    if not query:
        return
    results = requests.get(url=f"http://{OLLAMA_HOST}:7000/google/search?text={query}&limit=4", timeout=20)
    description = ' ,'.join([r['description'] for r in results.json()])
    return description, None
