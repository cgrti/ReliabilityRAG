"""
ReliabilityRAG Pipeline

Implements the full multi-stage pipeline:
1. Retrieval: Numpy-based vector search
2. Isolated Answering: LLM generates per-chunk provisional answers
3. Contradiction Graph: NLI-based pairwise contradiction detection
4. MIS/MWIS Filtering: Graph-based filtering of contradictory chunks
5. Final Generation: LLM produces reliable answer from clean set
"""
import time
from dataclasses import dataclass
from typing import Optional

from embedder import NumpyVectorSearch
from llm_engine import (
    LLMEngine,
    ISOLATED_ANSWER_PROMPT,
    FINAL_ANSWER_PROMPT,
    get_engine,
)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    company: str
    year: int
    section_type: str
    reliability_weight: float
    heading: str
    page_start: int
    score: float


class ReliabilityRAG:
    def __init__(self, top_k: int = 20, llm: Optional[LLMEngine] = None):
        self.top_k = top_k
        self._searcher = None
        self._nli = None
        self.llm = llm  # None = stub mode (no LLM)

    @property
    def searcher(self):
        if self._searcher is None:
            self._searcher = NumpyVectorSearch()
        return self._searcher

    @property
    def nli(self):
        if self._nli is None:
            from nli_graph import NLIContradictionGraph
            self._nli = NLIContradictionGraph()
        return self._nli

    # ── Stage 1: Retrieval ──────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        company_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k relevant chunks via numpy cosine search."""
        k = top_k or self.top_k
        results = self.searcher.search(query, top_k=k, company=company_filter, year=year_filter)

        return [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                text=r["text"],
                company=r["company"],
                year=r["year"],
                section_type=r["section_type"],
                reliability_weight=r["reliability_weight"],
                heading=r["heading"],
                page_start=r["page_start"],
                score=r["score"],
            )
            for r in results
        ]

    # ── Stage 2: Isolated Answering ─────────────────────────────────────

    def isolated_answer(self, query: str, chunk: RetrievedChunk) -> str:
        """Generate a provisional answer using ONLY this chunk."""
        if self.llm is None:
            return chunk.text  # Stub mode

        prompt = ISOLATED_ANSWER_PROMPT.format(
            company=chunk.company,
            year=chunk.year,
            section_type=chunk.section_type,
            context=chunk.text[:2000],
            question=query,
        )
        return self.llm.generate(prompt, max_tokens=200)

    def isolated_answer_all(self, query: str, chunks: list[RetrievedChunk]) -> list[str]:
        """Generate isolated answers for all chunks."""
        answers = []
        for i, chunk in enumerate(chunks):
            answer = self.isolated_answer(query, chunk)
            answers.append(answer)
            if self.llm and (i + 1) % 5 == 0:
                print(f"  Isolated answering: {i+1}/{len(chunks)}")
        return answers

    # ── Stage 5: Final Generation ───────────────────────────────────────

    def final_answer(
        self,
        query: str,
        clean_chunks: list[RetrievedChunk],
        max_sources: int = 5,
        source_chars: int = 600,
        max_tokens: int = None,
    ) -> str:
        """
        Generate final answer from clean (filtered) chunk set.

        Args:
            max_sources: How many chunks to include in LLM prompt (demo default: 5).
            source_chars: Truncation per source (demo default: 600 chars).
            max_tokens: Override max_tokens; else read from GPU profile.
        """
        # Build sources text
        source_parts = []
        for i, c in enumerate(clean_chunks[:max_sources]):
            source_parts.append(
                f"Kaynak {i+1} [{c.company} {c.year} | {c.section_type} | "
                f"güvenilirlik={c.reliability_weight}]:\n{c.text[:source_chars]}"
            )
        sources_text = "\n\n---\n\n".join(source_parts)

        if self.llm is None:
            return f"[LLM YOK — Ham kaynaklar]\n\nSorgu: {query}\n\n{sources_text}"

        if max_tokens is None:
            from config import get_profile
            max_tokens = get_profile()["final_answer_max_tokens"]

        prompt = FINAL_ANSWER_PROMPT.format(
            sources=sources_text,
            question=query,
        )
        return self.llm.generate(prompt, max_tokens=max_tokens)

    # ── Full Pipeline ───────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        company_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        use_nli_filter: bool = True,
        use_isolated_answering: bool = False,
    ) -> dict:
        """
        Run the full ReliabilityRAG pipeline.

        Args:
            use_isolated_answering: If True, generate per-chunk isolated answers
                via LLM before NLI (slow: ~O(top_k * LLM_latency)).
                If False (default), feed raw chunk text to NLI directly.
                Skipping is a common optimization for demo/real-time use.
        """
        timings = {}

        # Stage 1: Retrieval
        t0 = time.time()
        chunks = self.retrieve(question, top_k, company_filter, year_filter)
        timings["retrieval"] = time.time() - t0

        # Stage 2: Isolated Answering (optional — slow with LLM)
        t0 = time.time()
        if use_isolated_answering and self.llm is not None:
            isolated_answers = self.isolated_answer_all(question, chunks)
        else:
            # Feed raw chunk text to NLI — same behavior as stub mode
            isolated_answers = [c.text for c in chunks]
        timings["isolated_answering"] = time.time() - t0

        # Stage 3-4: NLI + Graph Filtering
        clean_chunks = chunks
        contradiction_edges = []

        if use_nli_filter and len(chunks) > 1:
            try:
                t0 = time.time()
                clean_indices, contradiction_edges = self.nli.filter_chunks(
                    isolated_answers,
                    [c.reliability_weight for c in chunks],
                    [c.year for c in chunks],
                    section_types=[c.section_type for c in chunks],
                    use_gat=True,
                )
                clean_chunks = [chunks[i] for i in clean_indices]
                timings["nli_graph_filter"] = time.time() - t0
            except Exception as e:
                print(f"NLI filter error: {e}")

        # Stage 5: Final Generation
        t0 = time.time()
        answer = self.final_answer(question, clean_chunks)
        timings["final_generation"] = time.time() - t0

        total_time = sum(timings.values())

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": len(chunks),
            "filtered_chunks": len(clean_chunks),
            "removed_chunks": len(chunks) - len(clean_chunks),
            "contradiction_edges": len(contradiction_edges),
            "timings": timings,
            "total_time": total_time,
            "chunks_detail": [
                {
                    "company": c.company,
                    "year": c.year,
                    "section_type": c.section_type,
                    "reliability": c.reliability_weight,
                    "score": round(c.score, 4),
                    # 600 chars matches final_answer source_chars; long enough
                    # for cached_demo.py to render a meaningful preview block.
                    "text_preview": c.text[:600],
                }
                for c in clean_chunks
            ],
        }


def demo():
    """Run demo queries with or without LLM."""
    import sys

    use_llm = "--llm" in sys.argv
    llm = None

    if use_llm:
        backend = "transformers"
        for arg in sys.argv:
            if arg.startswith("--backend="):
                backend = arg.split("=")[1]
        llm = get_engine(backend)

    rag = ReliabilityRAG(top_k=20, llm=llm)

    queries = [
        "Akbank'ın karbon emisyon azaltma hedefleri nelerdir?",
        "Garanti BBVA'nın toplam geliri ne kadar?",
        "Çimento şirketlerinin sürdürülebilirlik stratejileri nelerdir?",
    ]

    for q in queries:
        print(f"\n{'='*80}")
        print(f"SORU: {q}")
        print('='*80)

        result = rag.query(q, use_nli_filter=False)

        print(f"\nGetirilen: {result['retrieved_chunks']} | "
              f"Filtrelenen: {result['filtered_chunks']} | "
              f"Süre: {result['total_time']:.2f}s")

        print(f"\nCEVAP:\n{result['answer'][:500]}")


if __name__ == "__main__":
    demo()
