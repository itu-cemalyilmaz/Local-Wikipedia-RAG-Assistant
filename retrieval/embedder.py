"""
embedder.py
-----------
Wraps Ollama's nomic-embed-text model to produce local embeddings.
Uses only stdlib urllib — no openai, langchain, or sentence-transformers.

Also provides a ChromaDB-compatible EmbeddingFunction class so ChromaDB
can call this embedder transparently when querying or upserting.
"""

import json
import urllib.request
from typing import List

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"


import time

def embed_text(text: str, retries: int = 3) -> List[float]:
    """
    Generate an embedding vector for a single text string using Ollama.
    Retries on HTTP 500 errors to handle local memory spikes.
    """
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["embedding"]
        except urllib.error.HTTPError as e:
            if e.code == 500 and attempt < retries - 1:
                print(f"    [Ollama 500 Error] Truncating text to fit context window and retrying... (Attempt {attempt+1}/{retries})")
                text = text[:int(len(text) * 0.6)]  # Truncate to reduce tokens
                payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode("utf-8")
                req = urllib.request.Request(
                    url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
                )
                time.sleep(2)
                continue
            raise RuntimeError(f"Ollama API Error: {e}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                f"Make sure Ollama is running: `ollama serve`. Error: {e}"
            )

def embed_batch(texts: List[str], verbose: bool = False) -> List[List[float]]:
    """
    Embed a list of texts, one by one.
    """
    embeddings = []
    for i, text in enumerate(texts):
        if verbose and (i % 10 == 0):
            print(f"  Embedding {i+1}/{len(texts)} ...")
        embeddings.append(embed_text(text))
        time.sleep(0.05)  # Small sleep to prevent overwhelming Ollama
    return embeddings


from chromadb import EmbeddingFunction

class OllamaEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function using Ollama nomic-embed-text.

    ChromaDB expects an object with a __call__ method that accepts
    a list of strings and returns a list of embedding vectors.
    """
    def name(self) -> str:
        return "ollama_nomic_embed_text"

    def __call__(self, input: List[str]) -> List[List[float]]:
        return embed_batch(input, verbose=True)


if __name__ == "__main__":
    print(f"Testing embedding with model: {EMBED_MODEL}")
    try:
        vec = embed_text("Hello world")
        print(f"Embedding dimension: {len(vec)}")
        print(f"First 5 values: {vec[:5]}")
    except RuntimeError as e:
        print(f"Error: {e}")
