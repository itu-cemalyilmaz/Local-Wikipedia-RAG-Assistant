"""
llm.py
------
Handles answer generation using a local Ollama LLM via the /api/chat endpoint.
Uses only stdlib urllib — no openai or langchain dependencies.

RAG prompt design:
  - SYSTEM: strict grounding instructions + "I don't know" fallback
  - USER: [Context] + [Question]

The LLM is instructed to:
  1. Answer ONLY from the provided context
  2. Say "I don't know" when the answer is not in the context
  3. Cite entity names naturally (not chunk IDs)
"""

import json
import sys
import urllib.request
from typing import Generator, List, Optional

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions about famous people and famous places.

IMPORTANT RULES:
1. Answer ONLY using the information provided in the [Context] section below.
2. Do NOT use any prior knowledge or information not present in the context.
3. If the context does not contain enough information to answer the question, respond EXACTLY with: "I don't know based on the available information."
4. Keep answers clear, accurate, and concise (2-5 sentences unless more detail is needed).
5. When comparing two entities, address both clearly.
6. Do not make up facts, dates, or names.
"""


def build_rag_prompt(query: str, context_chunks: List[dict]) -> str:
    """
    Builds the user message content with context + question.

    Args:
        query: The user's question.
        context_chunks: List of retrieved chunk dicts {"text", "source", ...}.

    Returns:
        Formatted user message string.
    """
    if not context_chunks:
        context_str = "No relevant information found."
    else:
        # Deduplicate and format context
        seen_texts = set()
        context_parts = []
        for chunk in context_chunks:
            text = chunk["text"].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                source = chunk.get("source", "Unknown")
                context_parts.append(f"[Source: {source}]\n{text}")
        context_str = "\n\n---\n\n".join(context_parts)

    return f"""[Context]
{context_str}

[Question]
{query}"""


def generate_answer(
    query: str,
    context_chunks: List[dict],
    model: str = DEFAULT_MODEL,
    stream: bool = False,
) -> str:
    """
    Generates an answer using the local Ollama LLM given a query and context chunks.

    Args:
        query: The user's question.
        context_chunks: Retrieved chunks from the vector store.
        model: Ollama model name to use.
        stream: If True, streams the response (currently returns full text).

    Returns:
        The LLM's answer as a string.
    """
    user_message = build_rag_prompt(query, context_chunks)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,     # Low temp for factual grounding
            "num_predict": 512,     # Max output tokens
        },
    }

    url = f"{OLLAMA_BASE_URL}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["message"]["content"].strip()
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Make sure Ollama is running: `ollama serve`. Error: {e}"
        )


def generate_answer_streaming(
    query: str,
    context_chunks: List[dict],
    model: str = DEFAULT_MODEL,
) -> Generator[str, None, None]:
    """
    Streams the LLM response token by token.

    Args:
        query: The user's question.
        context_chunks: Retrieved chunks from the vector store.
        model: Ollama model name to use.

    Yields:
        String tokens as they are generated.
    """
    user_message = build_rag_prompt(query, context_chunks)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
        },
    }

    url = f"{OLLAMA_BASE_URL}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Make sure Ollama is running: `ollama serve`. Error: {e}"
        )


def check_ollama_available(model: str = DEFAULT_MODEL) -> tuple[bool, str]:
    """
    Checks if Ollama is running and the specified model is available.

    Returns:
        (is_available: bool, message: str)
    """
    try:
        url = f"{OLLAMA_BASE_URL}/api/tags"
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
            models = [m["name"] for m in result.get("models", [])]
            # Check if model name is a prefix match (e.g., "llama3.2:3b" or "llama3.2")
            model_available = any(m.startswith(model.split(":")[0]) for m in models)
            if model_available:
                return True, f"Ollama running. Model '{model}' available."
            else:
                return False, (
                    f"Ollama running but model '{model}' not found. "
                    f"Available: {models}. Run: ollama pull {model}"
                )
    except Exception as e:
        return False, f"Ollama not reachable at {OLLAMA_BASE_URL}. Run: ollama serve. Error: {e}"


if __name__ == "__main__":
    available, msg = check_ollama_available()
    print(f"Ollama status: {msg}")
    if available:
        test_chunks = [{"text": "Albert Einstein was a German-born physicist who developed the theory of relativity.", "source": "Albert Einstein"}]
        answer = generate_answer("Who was Albert Einstein?", test_chunks)
        print(f"\nAnswer: {answer}")
