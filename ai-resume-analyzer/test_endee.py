#!/usr/bin/env python3
"""
test_endee.py — End-to-end integration test

Tests the full pipeline directly (no FastAPI server needed):
  1. Connect to Endee
  2. Create index
  3. Insert vectors
  4. Search vectors
  5. Score results

Run:  python3 test_endee.py
Requires: Endee running on localhost:8080
"""

import sys
import json

# Add project to path
sys.path.insert(0, ".")

from embedding import generate_embedding, EMBEDDING_DIM
from endee_client import create_index, insert_vectors, search_vectors
from similarity import calculate_match_score, cosine_similarity

# ── Test data ────────────────────────────────────────────────────

RESUME = """
Experienced Python developer with 5 years of experience building
scalable backend systems. Proficient in FastAPI, Flask, and Django.
Deployed microservices on AWS using Docker and Kubernetes.
Built data pipelines using Apache Kafka and PostgreSQL.
Familiar with machine learning concepts and have implemented
recommendation engines using collaborative filtering.
Strong understanding of REST API design and database optimization.
"""

JOB_DESC = """
We are looking for a Senior Backend Engineer proficient in Python
with experience in cloud services, RESTful API design, and
containerization technologies like Docker.
"""

INDEX_NAME = "test_resume_index"


def main():
    print("=" * 60)
    print("  AI Resume Analyzer — End-to-End Test")
    print("=" * 60)
    passed = 0
    failed = 0

    # ── Test 1: Embedding generation ──
    print("\n[TEST 1] Generating embeddings …")
    try:
        resume_emb = generate_embedding(RESUME.strip())
        jd_emb = generate_embedding(JOB_DESC.strip())
        assert len(resume_emb) == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM}, got {len(resume_emb)}"
        assert len(jd_emb) == EMBEDDING_DIM
        assert resume_emb != jd_emb, "Different texts should produce different vectors"
        print(f"  ✅ PASS — Generated {EMBEDDING_DIM}-d embeddings (different texts → different vectors)")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1

    # ── Test 2: Cosine similarity ──
    print("\n[TEST 2] Cosine similarity …")
    try:
        self_sim = cosine_similarity(resume_emb, resume_emb)
        cross_sim = cosine_similarity(resume_emb, jd_emb)
        assert abs(self_sim - 1.0) < 0.001, f"Self-similarity should be 1.0, got {self_sim}"
        assert 0.0 <= cross_sim <= 1.0, f"Cross-similarity out of range: {cross_sim}"
        print(f"  ✅ PASS — self_sim={self_sim:.4f}, cross_sim={cross_sim:.4f}")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1

    # ── Test 3: Create Endee index ──
    print("\n[TEST 3] Creating Endee index …")
    try:
        resp = create_index(INDEX_NAME, dimension=EMBEDDING_DIM)
        if "error" in resp:
            print(f"  ❌ FAIL — {resp['error']}")
            print("  ⚠️  Is Endee running? → NDD_DATA_DIR=./data ./build/ndd")
            failed += 1
            print(f"\n{'=' * 60}")
            print(f"  Results: {passed} passed, {failed} failed (Endee offline)")
            print(f"{'=' * 60}")
            return
        assert resp["status"] in ("created", "exists")
        print(f"  ✅ PASS — index status: {resp['status']}")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1
        return

    # ── Test 4: Insert vectors ──
    print("\n[TEST 4] Inserting resume vector …")
    try:
        vectors = [{
            "id": "test_0",
            "vector": resume_emb,
            "meta": json.dumps({"text": RESUME.strip()[:100] + "…"}),
        }]
        resp = insert_vectors(INDEX_NAME, vectors)
        assert "error" not in resp, f"Insert error: {resp.get('error')}"
        print(f"  ✅ PASS — inserted {resp.get('count', '?')} vector(s)")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1

    # ── Test 5: Search vectors ──
    print("\n[TEST 5] Searching with job description embedding …")
    try:
        resp = search_vectors(INDEX_NAME, jd_emb, top_k=3)
        assert "error" not in resp, f"Search error: {resp.get('error')}"
        results = resp.get("results", [])
        assert len(results) > 0, "Expected at least 1 search result"
        print(f"  ✅ PASS — got {len(results)} result(s)")
        for i, r in enumerate(results):
            print(f"       [{i}] id={r['id']}, score={r['score']:.4f}")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1

    # ── Test 6: Match score ──
    print("\n[TEST 6] Computing weighted match score …")
    try:
        results = resp.get("results", [])
        score = calculate_match_score(results)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        print(f"  ✅ PASS — match_score = {score:.4f}")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1

    # ── Test 7: Full /analyze payload simulation ──
    print("\n[TEST 7] Full pipeline simulation …")
    try:
        # This mimics what POST /analyze does
        from main import chunk_text
        chunks = chunk_text(RESUME.strip())
        assert len(chunks) >= 1
        embeddings = [generate_embedding(c) for c in chunks]

        batch_vectors = [
            {"id": f"fulltest_{i}", "vector": emb, "meta": json.dumps({"text": c})}
            for i, (c, emb) in enumerate(zip(chunks, embeddings))
        ]
        insert_vectors(INDEX_NAME, batch_vectors)

        search_resp = search_vectors(INDEX_NAME, jd_emb, top_k=5)
        results = search_resp.get("results", [])
        final_score = calculate_match_score(results)

        top_sections = []
        for r in results:
            meta_raw = r.get("meta", "")
            if meta_raw:
                try:
                    meta = json.loads(meta_raw)
                    top_sections.append(meta.get("text", "?")[:60] + "…")
                except (json.JSONDecodeError, AttributeError):
                    top_sections.append(f"chunk_{r.get('id')}")

        assert 0.0 <= final_score <= 1.0
        assert len(top_sections) > 0

        print(f"  ✅ PASS — score={final_score:.4f}, sections={len(top_sections)}")
        for i, s in enumerate(top_sections):
            print(f"       [{i}] {s}")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1

    # ── Cleanup: delete the test index ──
    print("\n[CLEANUP] Deleting test index …")
    try:
        import requests
        r = requests.delete(f"http://localhost:8080/api/v1/index/{INDEX_NAME}/delete", timeout=5)
        print(f"  → {r.status_code}")
    except Exception:
        print("  → skipped")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    status = "ALL PASSED ✅" if failed == 0 else f"{failed} FAILED ❌"
    print(f"  Results: {passed} passed, {failed} failed — {status}")
    print(f"{'=' * 60}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
