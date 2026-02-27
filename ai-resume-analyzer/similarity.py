"""
similarity.py — Similarity Metrics

Provides cosine similarity calculation and helper utilities
for evaluating how closely resume sections match a job description.
"""

import math


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute the cosine similarity between two equal-length vectors.

    Cosine similarity = (A · B) / (‖A‖ × ‖B‖)

    Returns a value in [-1, 1], where 1 means identical direction.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score as a float.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimensionality")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude_1 = math.sqrt(sum(a * a for a in vec1))
    magnitude_2 = math.sqrt(sum(b * b for b in vec2))

    # Guard against zero-magnitude vectors
    if magnitude_1 == 0.0 or magnitude_2 == 0.0:
        return 0.0

    return dot_product / (magnitude_1 * magnitude_2)


def calculate_average_similarity(search_results: list[dict]) -> float:
    """
    Calculate the average similarity score from Endee search results.

    Each result dict is expected to contain a "score" key.

    Args:
        search_results: List of result dicts returned by Endee,
                        e.g. [{"id": 1, "score": 0.92}, ...].

    Returns:
        Mean similarity score, or 0.0 if no results.
    """
    if not search_results:
        return 0.0

    scores = [r.get("score", 0.0) for r in search_results]
    return sum(scores) / len(scores)
