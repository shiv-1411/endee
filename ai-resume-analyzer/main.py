"""
main.py — AI Resume Analyzer (FastAPI Backend)

Analyses a resume against a job description using vector embeddings
stored in the Endee vector database. Returns a semantic similarity
match score together with the top matching resume sections.

Run:
    uvicorn main:app --reload
"""

import json
import logging
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from embedding import generate_embedding, EMBEDDING_DIM
from endee_client import create_index, insert_vectors, search_vectors
from similarity import calculate_match_score

# ── Logging configuration ────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App initialisation ───────────────────────────────────────────

app = FastAPI(
    title="AI Resume Analyzer",
    description="Semantic resume-to-job matching powered by Endee vector DB",
    version="1.0.0",
)

# Name of the Endee index used for resume vectors
INDEX_NAME = "resume_index"


# ── Request / Response schemas ───────────────────────────────────


class AnalyzeRequest(BaseModel):
    """Incoming payload for the /analyze endpoint."""
    resume_text: str = Field(..., min_length=1, description="Full resume text")
    job_description: str = Field(..., min_length=1, description="Target job description")


class AnalyzeResponse(BaseModel):
    """Response returned by the /analyze endpoint."""
    match_score: float
    top_matching_sections: list[str]
    message: str


# ── Helper utilities ─────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    """
    Split text into roughly equal-sized word chunks.

    Each chunk contains at most ``chunk_size`` words.  If the input
    is shorter than one chunk, it is returned as a single-element list.

    Args:
        text:       The input text to split.
        chunk_size: Maximum number of words per chunk.

    Returns:
        List of non-empty text chunks.
    """
    words = text.split()
    chunks: list[str] = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks if chunks else [text]


# ── Endpoints ────────────────────────────────────────────────────


@app.get("/")
def health_check():
    """Simple health-check endpoint."""
    return {"status": "ok", "message": "AI Resume Analyzer is running"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_resume(payload: AnalyzeRequest):
    """
    Analyse a resume against a job description.

    Workflow:
      1. Split the resume into text chunks.
      2. Generate an embedding for each chunk.
      3. Create an Endee index (idempotent — safe to call repeatedly).
      4. Insert resume chunk vectors into Endee.
      5. Embed the job description.
      6. Search Endee for the most similar resume chunks.
      7. Compute a weighted match score and return results.
    """

    # ── Step 1: chunk the resume ──
    logger.info("Step 1/7 — Chunking resume text …")
    chunks = chunk_text(payload.resume_text)
    logger.info("  → %d chunk(s) produced.", len(chunks))

    # ── Step 2: embed each chunk ──
    logger.info("Step 2/7 — Generating embeddings for %d chunk(s) …", len(chunks))
    chunk_embeddings = [generate_embedding(c) for c in chunks]

    # ── Step 3: create Endee index (ignored if it already exists) ──
    logger.info("Step 3/7 — Ensuring Endee index '%s' exists …", INDEX_NAME)
    dimension = len(chunk_embeddings[0])
    idx_resp = create_index(INDEX_NAME, dimension=dimension)
    if "error" in idx_resp:
        logger.warning("Index creation returned error: %s", idx_resp["error"])

    # ── Step 4: insert chunk vectors into Endee ──
    logger.info("Step 4/7 — Inserting chunk vectors into Endee …")

    # Use a unique batch prefix so repeated calls don't collide on IDs
    batch_id = uuid.uuid4().hex[:8]
    vectors_payload = [
        {
            "id": f"{batch_id}_{idx}",
            "vector": emb,
            "meta": json.dumps({"text": chunk}),
        }
        for idx, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings))
    ]
    ins_resp = insert_vectors(INDEX_NAME, vectors_payload)
    if "error" in ins_resp:
        logger.warning("Vector insertion returned error: %s", ins_resp["error"])

    # ── Step 5: embed the job description ──
    logger.info("Step 5/7 — Embedding the job description …")
    jd_embedding = generate_embedding(payload.job_description)

    # ── Step 6: search for nearest resume chunks ──
    logger.info("Step 6/7 — Searching Endee for matching chunks …")
    search_response = search_vectors(INDEX_NAME, jd_embedding, top_k=5)

    # Extract results list from the response
    results = search_response.get("results", [])

    # ── Step 7: calculate score & build response ──
    logger.info("Step 7/7 — Computing weighted match score …")
    match_score = calculate_match_score(results)

    # Collect the text of the top-matching chunks (from meta)
    top_sections: list[str] = []
    for r in results:
        meta_raw = r.get("meta", "")
        if meta_raw:
            try:
                meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
                top_sections.append(meta.get("text", f"chunk_{r.get('id', '?')}"))
            except (json.JSONDecodeError, AttributeError):
                top_sections.append(f"chunk_{r.get('id', '?')}")
        else:
            top_sections.append(f"chunk_{r.get('id', '?')}")

    logger.info("✔ Analysis complete — match_score=%.4f, sections=%d",
                match_score, len(top_sections))

    return AnalyzeResponse(
        match_score=round(match_score, 4),
        top_matching_sections=top_sections,
        message="Basic semantic similarity calculated",
    )
