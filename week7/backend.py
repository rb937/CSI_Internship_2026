"""
backend.py
Provider-agnostic interface for LLM generation and embeddings.
Default provider is Ollama (local, no API key needed). OpenAI and Anthropic
paths are stubbed and ready to use once you have API keys to test with.

Swap providers by changing LLM_PROVIDER / EMBED_PROVIDER below, or by
passing provider="openai" / "anthropic" directly to the functions.
"""

import os
import requests

# ---- Config ----
LLM_PROVIDER = "ollama"          # "ollama" | "openai" | "anthropic"
EMBED_PROVIDER = "ollama"        # "ollama" | "openai"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "qwen3.5:9b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

OPENAI_LLM_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
ANTHROPIC_LLM_MODEL = "claude-sonnet-4-6"


# ---------------- LLM GENERATION ----------------

def get_llm_response(prompt: str, provider: str = None) -> str:
    """Generate a response from the configured (or specified) LLM provider."""
    provider = provider or LLM_PROVIDER

    if provider == "ollama":
        return _ollama_generate(prompt)
    elif provider == "openai":
        return _openai_generate(prompt)
    elif provider == "anthropic":
        return _anthropic_generate(prompt)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _ollama_generate(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_LLM_MODEL, "prompt": prompt, "think":False, "stream": False},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def _openai_generate(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content.strip()


def _anthropic_generate(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=ANTHROPIC_LLM_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


# ---------------- EMBEDDINGS ----------------

def get_embedding(text: str, provider: str = None) -> list:
    """Return an embedding vector for the given text."""
    provider = provider or EMBED_PROVIDER

    if provider == "ollama":
        return _ollama_embed(text)
    elif provider == "openai":
        return _openai_embed(text)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def _ollama_embed(text: str) -> list:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def _openai_embed(text: str) -> list:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text)
    return response.data[0].embedding
