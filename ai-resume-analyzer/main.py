"""
main.py — AI Resume Analyzer (FastAPI Backend)

Analyses a resume against a job description using vector embeddings
stored in the Endee vector database. Returns a semantic similarity
match score together with the top matching resume sections.

Run:
    uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embedding import generate_embedding
from endee_client import create_index, insert_vectors, search_vectors
from similarity import calculate_average_similarity

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
    resume_text: str
    job_description: str


class AnalyzeResponse(BaseModel):
    """Response returned by the /analyze endpoint."""
    match_score: float
    top_matching_sections: list[str]
    message: str


# ── Helper utilities ─────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    """
    Split text into roughly equal-sized word chunks.

    Each chunk contains at most `chunk_size` words.
    """
    words = text.split()
    chunks: list[str] = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
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
      3. Create an Endee index (idempotent).
      4. Insert resume chunk vectors into Endee.
      5. Embed the job description.
      6. Search Endee for the most similar resume chunks.
      7. Compute an average similarity score and return results.
    """

    # ── Step 1: chunk the resume ──
    chunks = chunk_text(payload.resume_text)

    # ── Step 2: embed each chunk ──
    chunk_embeddings = [generate_embedding(c) for c in chunks]

    # ── Step 3: create Endee index (ignored if it already exists) ──
    dimension = len(chunk_embeddings[0])
    create_index(INDEX_NAME, dimension=dimension)

    # ── Step 4: insert chunk vectors into Endee ──
    vectors_payload = [
        {
            "id": idx,
            "vector": emb,
            "metadata": {"text": chunk},
        }
        for idx, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings))
    ]
    insert_vectors(INDEX_NAME, vectors_payload)

    # ── Step 5: embed the job description ──
    jd_embedding = generate_embedding(payload.job_description)

    # ── Step 6: search for nearest resume chunks ──
    search_response = search_vectors(INDEX_NAME, jd_embedding, top_k=5)

    # Extract results list from the response
    results = search_response.get("results", [])

    # ── Step 7: calculate score & build response ──
    match_score = calculate_average_similarity(results)

    # Collect the text of the top-matching chunks
    top_sections = [
        r.get("metadata", {}).get("text", f"chunk_{r.get('id', '?')}")
        for r in results
    ]

    return AnalyzeResponse(
        match_score=round(match_score, 4),
        top_matching_sections=top_sections,
        message="Basic semantic similarity calculated",
    )
