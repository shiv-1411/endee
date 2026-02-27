"""
endee_client.py — Endee Vector Database Client

Thin wrapper around the Endee REST API for index management
and vector insert / search operations.
"""

import requests

# Endee server base URL (assumes local instance)
BASE_URL = "http://localhost:8080"


# ── Index Management ─────────────────────────────────────────────


def create_index(index_name: str, dimension: int = 384) -> dict:
    """
    Create a new vector index in Endee.

    Args:
        index_name: Unique name for the index.
        dimension:  Dimensionality of vectors to be stored.

    Returns:
        JSON response from Endee.
    """
    url = f"{BASE_URL}/api/v1/index/create"

    payload = {
        "index_name": index_name,
        "dimension": dimension,
        "metric": "cosine",  # cosine similarity is ideal for text embeddings
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        return {"error": str(exc)}


# ── Vector Operations ────────────────────────────────────────────


def insert_vectors(index_name: str, vectors: list[dict]) -> dict:
    """
    Insert one or more vectors into an existing Endee index.

    Args:
        index_name: Target index name.
        vectors:    List of dicts, each containing:
                      - id   (int)   : unique vector id
                      - vector (list) : the embedding values
                      - metadata (dict, optional) : extra payload

    Returns:
        JSON response from Endee.
    """
    url = f"{BASE_URL}/api/v1/vector/insert"

    payload = {
        "index_name": index_name,
        "vectors": vectors,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        return {"error": str(exc)}


def search_vectors(
    index_name: str,
    query_vector: list[float],
    top_k: int = 5,
) -> dict:
    """
    Search for the nearest vectors in Endee.

    Args:
        index_name:   Index to search.
        query_vector: The query embedding.
        top_k:        Number of nearest neighbours to return.

    Returns:
        JSON response from Endee containing ranked results.
    """
    url = f"{BASE_URL}/api/v1/vector/search"

    payload = {
        "index_name": index_name,
        "vector": query_vector,
        "top_k": top_k,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        return {"error": str(exc)}
