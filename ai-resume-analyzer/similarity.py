"""
similarity.py — Similarity Metrics

Provides cosine similarity calculation and helper utilities
for evaluating how closely resume sections match a job description.
"""

import math
import logging

logger = logging.getLogger(__name__)


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


def calculate_match_score(search_results: list[dict]) -> float:
    """
    Calculate a weighted match score from Endee search results.

    Higher-ranked results (those returned first) receive more weight
    because the closest matches matter more than marginal ones.

    Weighting scheme:
        weight_i = (n - i) / sum(1..n)
        → first result gets weight n/Σ, last gets weight 1/Σ

    Each result dict is expected to have a "score" key (distance or
    similarity returned by Endee).

    Args:
        search_results: Ranked list of result dicts from Endee,
                        e.g. [{"id": "0", "score": 0.92}, ...].

    Returns:
        Weighted match score in [0, 1], or 0.0 if no results.
    """
    if not search_results:
        logger.warning("No search results to score.")
        return 0.0

    n = len(search_results)
    # Sum of arithmetic series 1+2+…+n for normalisation
    weight_sum = n * (n + 1) / 2

    weighted_total = 0.0
    for rank, result in enumerate(search_results):
        score = result.get("score", 0.0)
        weight = (n - rank) / weight_sum  # higher weight for top matches
        weighted_total += score * weight

    final = min(max(weighted_total, 0.0), 1.0)  # clamp to [0, 1]
    logger.info("Weighted match score: %.4f (from %d results)", final, n)
    return final
