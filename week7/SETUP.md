# Provider Setup

This RAG system is provider-agnostic — the LLM and embedding model used are
controlled entirely from `backend.py`. By default it runs fully local via
**Ollama** (no API key, no cost, works offline), but can be switched to
OpenAI or Anthropic if you don't want to run a local model.

## Option 1: Local (default) — Ollama

Requires [Ollama](https://ollama.com) installed and running locally. taking `qwen3.5:9b` and `nomic-embed-text` for example.

```bash
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

No changes needed — `backend.py` already defaults to:
```python
LLM_PROVIDER = "ollama"
EMBED_PROVIDER = "ollama"
```

## Option 2: OpenAI

1. Install the SDK: `pip install openai`
2. Set your API key as an environment variable:
   ```bash
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="your-key-here"

   # macOS/Linux
   export OPENAI_API_KEY="your-key-here"
   ```
3. In `backend.py`, change:
   ```python
   LLM_PROVIDER = "openai"
   EMBED_PROVIDER = "openai"
   ```

## Option 3: Anthropic (LLM only — no embedding endpoint)

1. Install the SDK: `pip install anthropic`
2. Set your API key:
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   ```
3. In `backend.py`, change:
   ```python
   LLM_PROVIDER = "anthropic"
   EMBED_PROVIDER = "openai"  # Anthropic has no embeddings API — pair with OpenAI or keep Ollama for embeddings
   ```

## Mixing providers

You don't have to match LLM and embedding providers. For example, local
embeddings (free, fast) with a hosted LLM for generation:
```python
LLM_PROVIDER = "openai"
EMBED_PROVIDER = "ollama"
```

## Per-call override

Instead of changing the global config, you can also pass `provider=` directly:
```python
from chatbot import answer_question

answer_question("What is this document about?", provider="openai")
```

## Notes

- Switching `EMBED_PROVIDER` after documents are already indexed will cause a
  mismatch (embeddings from different models aren't comparable) — re-run
  ingestion after changing embedding provider.
- OpenAI and Anthropic paths are implemented but only lightly tested — if you
  hit an error, check the exact response format matches the SDK version
  you have installed.
