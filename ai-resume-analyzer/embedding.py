"""
embedding.py — Text Embedding Generator

Generates vector embeddings for text input.
Uses OpenAI's embedding API when an API key is available,
otherwise falls back to a deterministic hash-based dummy vector.
"""

import os
import hashlib
import logging

logger = logging.getLogger(__name__)

# Dimension of the embedding vectors (matches text-embedding-3-small output)
EMBEDDING_DIM = 384


def generate_embedding(text: str) -> list[float]:
    """
    Generate a vector embedding for the given text.

    Strategy:
      1. If the OPENAI_API_KEY environment variable is set,
         call the OpenAI Embeddings API (text-embedding-3-small).
      2. Otherwise, produce a deterministic dummy vector derived
         from the SHA-256 hash of the input text.

    Args:
        text: The input string to embed.

    Returns:
        A list of floats representing the embedding vector (384-d).
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        logger.info("Generating OpenAI embedding (text-embedding-3-small) …")
        return _openai_embedding(text, api_key)
    else:
        logger.info("No OPENAI_API_KEY set — using deterministic dummy embedding.")
        return _dummy_embedding(text)


# ── Private helpers ──────────────────────────────────────────────


def _openai_embedding(text: str, api_key: str) -> list[float]:
    """
    Call OpenAI Embeddings API and return the vector.

    Uses the text-embedding-3-small model which produces a 1536-d
    vector by default. We request 384 dimensions for efficiency.
    """
    import openai

    client = openai.OpenAI(api_key=api_key)

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
        dimensions=EMBEDDING_DIM,
    )

    return response.data[0].embedding


def _dummy_embedding(text: str) -> list[float]:
    """
    Generate a deterministic dummy embedding from the text hash.

    The vector is reproducible for identical inputs, which allows
    the similarity pipeline to work end-to-end without an API key.
    Different texts produce different vectors, so cosine similarity
    still varies across inputs.
    """
    # Use SHA-256 to get a stable hash regardless of Python session
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Convert the first 8 hex chars to a seed value
    seed = int(digest[:8], 16)

    # Build a 384-dim vector with values in [0, 1)
    vector: list[float] = []
    for i in range(EMBEDDING_DIM):
        # Simple deterministic spread using the seed and index
        val = ((seed * (i + 1)) % 1_000_000) / 1_000_000.0
        vector.append(val)

    return vector
