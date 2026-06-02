"""
Embedding Pipeline for ReliabilityRAG

Two-phase approach:
1. Compute embeddings with sentence-transformers → save as .npy
2. Load from disk for fast numpy-based cosine search (no external DB needed)

ChromaDB insertion is available as an optional step.
"""
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import CHUNKS_OUTPUT, VECTOR_DB_PATH, PROJECT_ROOT

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
BATCH_SIZE = 128

# Paths for cached embeddings
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_NPY = EMBEDDINGS_DIR / "embeddings.npy"
CHUNK_IDS_JSON = EMBEDDINGS_DIR / "chunk_ids.json"


def load_chunks(path: Path = CHUNKS_OUTPUT) -> list[dict]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks")
    return chunks


# ── Phase 1: Compute & Save Embeddings ─────────────────────────────────

def compute_and_save_embeddings(chunks: list[dict]):
    """Compute embeddings and save to disk as numpy array."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [f"passage: {c['text'][:8000]}" for c in chunks]

    print(f"Generating embeddings for {len(texts)} chunks (batch_size={BATCH_SIZE})...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({len(texts)/elapsed:.0f} chunks/sec)")

    # Save embeddings as numpy
    np.save(str(EMBEDDINGS_NPY), embeddings)
    print(f"Saved embeddings: {EMBEDDINGS_NPY} ({embeddings.shape})")

    # Save chunk IDs for index mapping
    chunk_ids = [c["chunk_id"] for c in chunks]
    with open(CHUNK_IDS_JSON, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)
    print(f"Saved chunk IDs: {CHUNK_IDS_JSON}")

    return embeddings


# ── Phase 2: Fast Numpy Search ──────────────────────────────────────────

class NumpyVectorSearch:
    """Fast cosine similarity search using numpy. No external DB needed.

    Performance optimizations (2026-05-31, latency target ≤5s):
      1. Metadata arrays pre-computed at init → vectorized boolean filter
         instead of 182K-iteration Python loop (was ~3-5s per filtered query).
      2. SentenceTransformer model eagerly loaded + moved to GPU if available
         → eliminates 5s cold-load on first search() call.
      3. argpartition instead of argsort for top-k → O(N) vs O(N log N).
    """

    def __init__(self, eager_load_model: bool = True):
        print("Loading embeddings from disk...")
        self.embeddings = np.load(str(EMBEDDINGS_NPY))
        with open(CHUNK_IDS_JSON, "r") as f:
            self.chunk_ids = json.load(f)
        self.chunks = load_chunks()
        self.chunk_map = {c["chunk_id"]: c for c in self.chunks}

        # Pre-compute metadata arrays for vectorized filtering (huge win:
        # replaces 182K-element Python for-loop with NumPy boolean indexing,
        # ~50-100x speedup on filtered queries).
        self._companies_arr = np.array([c["company"] for c in self.chunks])
        self._years_arr = np.array([c["year"] for c in self.chunks])

        self._model = None
        print(f"Ready: {self.embeddings.shape[0]} vectors, dim={self.embeddings.shape[1]}")

        # Eager model load — moves the 5s cold-start cost from first search()
        # to startup, where it's expected. Critical for ≤5s latency target.
        if eager_load_model:
            _ = self.model

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            # Try to move to GPU if available (encoding 1 query is fast on
            # CPU too, but GPU is consistent and avoids cold-cache jitter).
            try:
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.to("cuda")
            except Exception:
                pass
        return self._model

    def search(self, query: str, top_k: int = 20, company: str = None, year: int = None) -> list[dict]:
        """Search for most similar chunks. Vectorized filtering."""
        # Encode query (model already warm + on GPU if available)
        query_emb = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Cosine similarity (embeddings are already normalized)
        scores = (self.embeddings @ query_emb.T).flatten()

        # Vectorized metadata filter (was: Python for-loop over 182K)
        if company:
            scores[self._companies_arr != company] = -np.inf
        if year:
            scores[self._years_arr != year] = -np.inf

        # argpartition is O(N) vs argsort O(N log N); for 182K vectors this
        # saves a few hundred ms on hot calls.
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            part = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = part[np.argsort(scores[part])[::-1]]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0 or scores[idx] == -np.inf:
                break
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "company": chunk["company"],
                "year": chunk["year"],
                "section_type": chunk["section_type"],
                "reliability_weight": chunk["reliability_weight"],
                "heading": chunk["heading"],
                "page_start": chunk["page_start"],
                "score": float(scores[idx]),
            })

        return results


# ── Optional: ChromaDB Loading ──────────────────────────────────────────

def load_into_chromadb(chunks: list[dict], embeddings_path: Path = EMBEDDINGS_NPY):
    """Load pre-computed embeddings into ChromaDB (optional)."""
    import chromadb

    embeddings = np.load(str(embeddings_path))
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    try:
        client.delete_collection("integrated_reports")
    except Exception:
        pass

    collection = client.create_collection(
        name="integrated_reports",
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 2000
    for start in tqdm(range(0, len(chunks), BATCH), desc="ChromaDB insert"):
        end = min(start + BATCH, len(chunks))
        batch = chunks[start:end]
        batch_emb = embeddings[start:end].tolist()

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=batch_emb,
            metadatas=[{
                "company": c["company"],
                "year": c["year"],
                "heading": c["heading"][:500],
                "section_type": c["section_type"],
                "reliability_weight": c["reliability_weight"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "has_table": c["has_table"],
                "source_folder": c["source_folder"],
            } for c in batch],
        )

    print(f"ChromaDB: {collection.count()} vectors stored")


# ── Test ────────────────────────────────────────────────────────────────

def test_search():
    """Run sample queries."""
    searcher = NumpyVectorSearch()

    queries = [
        "Akbank'ın karbon emisyon azaltma hedefleri nelerdir?",
        "Garanti BBVA'nın toplam geliri ne kadar?",
        "Çimento şirketlerinin sürdürülebilirlik stratejileri",
        "İş sağlığı ve güvenliği kazaları",
        "Yenilenebilir enerji yatırımları",
    ]

    for q in queries:
        print(f"\n{'='*70}")
        print(f"SORU: {q}")
        print('='*70)
        results = searcher.search(q, top_k=5)
        for i, r in enumerate(results):
            print(f"  {i+1}. [{r['company']} {r['year']}] "
                  f"({r['section_type']}, sim={r['score']:.3f})")
            print(f"     {r['text'][:120]}...")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_search()
    elif len(sys.argv) > 1 and sys.argv[1] == "chromadb":
        chunks = load_chunks()
        load_into_chromadb(chunks)
    else:
        chunks = load_chunks()
        compute_and_save_embeddings(chunks)
        print("\n--- Running test search ---")
        test_search()
