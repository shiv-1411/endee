# AI Resume Analyzer — Powered by Endee Vector Database

A minimal, backend-only application that performs **semantic similarity matching** between a candidate's resume and a job description using vector embeddings stored in the [Endee](https://github.com/endee-project/endee) vector database.

---

## Project Overview

Traditional keyword-based resume screening misses contextual meaning — a resume that says *"built scalable microservices"* won't match a job posting asking for *"distributed systems experience"* even though they are semantically related.

This project solves that problem by converting text into **vector embeddings** and using **cosine similarity search** inside Endee to find the most relevant resume sections for a given job description.

---

## Problem Statement

Recruiters and hiring pipelines rely on keyword matching, which:

- Fails to capture semantic meaning.
- Penalises candidates who use different terminology.
- Cannot rank partial matches.

**Goal:** Build a lightweight tool that scores resume–job alignment using vector similarity rather than keyword overlap.

---

## Solution Approach

1. **Chunk** the resume into manageable text segments.
2. **Embed** each chunk into a high-dimensional vector (384-d by default).
3. **Store** the vectors in an Endee index.
4. **Embed** the job description into the same vector space.
5. **Search** Endee for the resume chunks closest to the job description.
6. **Score** the match by averaging the similarity of the top results.

> When an OpenAI API key is provided, real embeddings from `text-embedding-3-small` are used. Without one, the system falls back to deterministic hash-based dummy vectors so the pipeline still runs end-to-end.

---

## System Architecture

```
┌──────────────┐         ┌────────────────┐
│   Client     │  POST   │  FastAPI       │
│  (curl/etc.) │────────▶│  /analyze      │
└──────────────┘         └──────┬─────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
        ┌───────────┐   ┌─────────────┐    ┌────────────┐
        │ embedding │   │ endee_client│    │ similarity │
        │  .py      │   │  .py        │    │  .py       │
        └─────┬─────┘   └──────┬──────┘    └────────────┘
              │                │
              ▼                ▼
     ┌──────────────┐  ┌──────────────┐
     │ OpenAI API   │  │ Endee Vector │
     │ (optional)   │  │ Database     │
     └──────────────┘  └──────────────┘
```

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app, endpoints, orchestration logic |
| `embedding.py` | Text → vector embedding (OpenAI or fallback) |
| `endee_client.py` | REST client for Endee (create, insert, search) with MsgPack decoding |
| `similarity.py` | Cosine similarity & score aggregation |

---

## How Endee Is Used

Endee serves as the **vector storage and retrieval engine**:

| Operation | Endee Endpoint | Purpose |
|---|---|---|
| Create Index | `POST /api/v1/index/create` | Initialise an index with a given dimensionality and cosine metric |
| Insert Vectors | `POST /api/v1/index/<name>/vector/insert` | Store resume chunk embeddings with metadata |
| Search Vectors | `POST /api/v1/index/<name>/search` | Find the *k* nearest resume chunks to the job description embedding |

Search results are returned in **MessagePack** format and decoded automatically by the client.

All communication happens over HTTP via the `requests` + `msgpack` libraries — no native drivers or SDKs required.

---

## API Endpoints

### `GET /`

Health check.

**Response:**
```json
{ "status": "ok", "message": "AI Resume Analyzer is running" }
```

### `POST /analyze`

Analyse a resume against a job description.

**Request body:**
```json
{
  "resume_text": "Experienced backend engineer with 5 years ...",
  "job_description": "Looking for a senior Python developer ..."
}
```

**Response:**
```json
{
  "match_score": 0.8723,
  "top_matching_sections": [
    "Experienced backend engineer with 5 years ...",
    "Built REST APIs using FastAPI and Flask ..."
  ],
  "message": "Basic semantic similarity calculated"
}
```

---

## Setup Instructions

### Prerequisites

- **Python 3.10+**
- **Endee** vector database running locally on port `8080`

### 1. Clone & navigate

```bash
git clone <your-fork-url>
cd endee/ai-resume-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Set OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

> If omitted, the system uses deterministic dummy embeddings — the full pipeline still works.

### 5. Start the server

```bash
uvicorn main:app --reload
```

### 6. Test

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Python developer with experience in FastAPI, Docker, and AWS.",
    "job_description": "Senior backend engineer proficient in Python and cloud services."
  }'
```

---

## Future Improvements

- **PDF / DOCX parsing** — accept file uploads instead of raw text.
- **Persistent index management** — reuse indexes across requests and support multiple candidates.
- **Real-time embedding models** — integrate open-source models (e.g., Sentence-Transformers) for offline use.
- **Weighted section scoring** — give higher weight to skills and experience sections.
- **Batch analysis** — compare multiple resumes against one job description in a single call.
- **Frontend dashboard** — visualise match scores and highlight relevant resume sections.

---

## License

This project is part of the Endee repository and follows its license terms.
