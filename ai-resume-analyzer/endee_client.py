"""
endee_client.py — Endee Vector Database Client

Thin wrapper around the Endee REST API for index management
and vector insert / search operations.

API reference (from src/main.cpp):
  POST /api/v1/index/create              → create a new index
  POST /api/v1/index/<name>/vector/insert → insert vectors (JSON or MsgPack)
  POST /api/v1/index/<name>/search       → kNN search (returns MsgPack)
"""

import json
import logging
import requests
import msgpack

logger = logging.getLogger(__name__)

# Endee server base URL (assumes local instance)
BASE_URL = "http://localhost:8080"


# ── Index Management ─────────────────────────────────────────────


def create_index(index_name: str, dimension: int = 384) -> dict:
    """
    Create a new vector index in Endee.

    If the index already exists Endee returns HTTP 409 — this is
    treated as a success so the call is idempotent.

    Args:
        index_name: Unique name for the index.
        dimension:  Dimensionality of vectors to be stored.

    Returns:
        dict with "status" key ("created" or "exists") on success,
        or an "error" key on failure.
    """
    url = f"{BASE_URL}/api/v1/index/create"

    # Endee expects "dim" and "space_type" (not "dimension" / "metric")
    payload = {
        "index_name": index_name,
        "dim": dimension,
        "space_type": "cosine",  # cosine similarity for text embeddings
    }

    try:
        logger.info("Creating Endee index '%s' (dim=%d) …", index_name, dimension)
        resp = requests.post(url, json=payload, timeout=10)

        if resp.status_code == 409:
            # Index already exists — perfectly fine
            logger.info("Index '%s' already exists, reusing.", index_name)
            return {"status": "exists"}

        resp.raise_for_status()
        logger.info("Index '%s' created successfully.", index_name)
        return {"status": "created"}

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Endee at %s", BASE_URL)
        return {"error": f"Cannot connect to Endee at {BASE_URL}"}
    except requests.exceptions.RequestException as exc:
        logger.error("Index creation failed: %s", exc)
        return {"error": str(exc)}


# ── Vector Operations ────────────────────────────────────────────


def insert_vectors(index_name: str, vectors: list[dict]) -> dict:
    """
    Insert one or more vectors into an existing Endee index.

    Args:
        index_name: Target index name.
        vectors:    List of dicts, each containing:
                      - id     (str)  : unique vector id
                      - vector (list) : the embedding values
                      - meta   (str, optional) : JSON-encoded metadata

    Returns:
        dict with "status" on success, or "error" on failure.
    """
    # Endee URL pattern: /api/v1/index/<name>/vector/insert
    url = f"{BASE_URL}/api/v1/index/{index_name}/vector/insert"

    # Endee expects a JSON array at the top level (not wrapped in an object)
    payload = []
    for v in vectors:
        entry = {
            "id": str(v["id"]),           # Endee accepts string IDs
            "vector": v["vector"],
        }
        # Endee stores metadata in a "meta" field as a JSON string
        if "meta" in v and v["meta"]:
            entry["meta"] = v["meta"] if isinstance(v["meta"], str) else json.dumps(v["meta"])
        payload.append(entry)

    try:
        logger.info("Inserting %d vector(s) into index '%s' …", len(payload), index_name)
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        logger.info("Inserted %d vector(s) successfully.", len(payload))
        return {"status": "ok", "count": len(payload)}

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Endee at %s", BASE_URL)
        return {"error": f"Cannot connect to Endee at {BASE_URL}"}
    except requests.exceptions.RequestException as exc:
        logger.error("Vector insertion failed: %s", exc)
        return {"error": str(exc)}


def search_vectors(
    index_name: str,
    query_vector: list[float],
    top_k: int = 5,
) -> dict:
    """
    Search for the nearest vectors in Endee.

    Endee returns search results as MessagePack, so this function
    deserialises the response before returning a Python dict.

    Args:
        index_name:   Index to search.
        query_vector: The query embedding.
        top_k:        Number of nearest neighbours to return.

    Returns:
        dict with a "results" list on success, or "error" on failure.
        Each result contains "id", "score", and optionally "meta".
    """
    # Endee URL pattern: /api/v1/index/<name>/search
    url = f"{BASE_URL}/api/v1/index/{index_name}/search"

    # Endee expects "k" (not "top_k")
    payload = {
        "vector": query_vector,
        "k": top_k,
    }

    try:
        logger.info("Searching index '%s' for top-%d matches …", index_name, top_k)
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()

        # Endee returns MessagePack-encoded results
        content_type = resp.headers.get("Content-Type", "")

        if "msgpack" in content_type:
            raw = msgpack.unpackb(resp.content, raw=False)
        else:
            # Fallback to JSON if the server returns JSON
            raw = resp.json()

        # Normalise into a consistent {"results": [...]} format
        results = _normalise_search_results(raw)
        logger.info("Search returned %d result(s).", len(results))
        return {"results": results}

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Endee at %s", BASE_URL)
        return {"error": f"Cannot connect to Endee at {BASE_URL}", "results": []}
    except requests.exceptions.RequestException as exc:
        logger.error("Vector search failed: %s", exc)
        return {"error": str(exc), "results": []}


# ── Helpers ──────────────────────────────────────────────────────


def _normalise_search_results(raw) -> list[dict]:
    """
    Convert the raw Endee search response (MsgPack-decoded) into a
    flat list of {id, score, meta} dicts.

    Endee's ResultSet is serialised via MsgPack and may arrive as a
    list of [id, distance] pairs, a dict with various keys, etc.
    This function handles the common shapes gracefully.
    """
    results: list[dict] = []

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                results.append({
                    "id": item.get("id", ""),
                    "score": float(item.get("distance", item.get("score", 0.0))),
                    "meta": item.get("meta", ""),
                })
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # [id, distance] pair
                results.append({
                    "id": str(item[0]),
                    "score": float(item[1]),
                    "meta": "",
                })
    elif isinstance(raw, dict):
        # Might be wrapped: {"results": [...]} or {"ids":[], "distances":[]}
        if "results" in raw:
            return _normalise_search_results(raw["results"])
        ids = raw.get("ids", [])
        distances = raw.get("distances", [])
        for i, (vid, dist) in enumerate(zip(ids, distances)):
            results.append({"id": str(vid), "score": float(dist), "meta": ""})

    return results

