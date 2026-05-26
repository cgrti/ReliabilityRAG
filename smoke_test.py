"""
3-query end-to-end smoke test: Retrieval -> NLI -> GAT+MWIS -> Final Generation.
Isolated answering is skipped (raw chunks -> NLI) for demo speed.
"""
import sys
import time

# Force UTF-8 output so Turkish + any unicode (em-dashes, box-drawing, etc.)
# survive the print path on Windows cp1254 consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from rag_pipeline import ReliabilityRAG
from llm_engine import get_engine


def main():
    # Company names must match metadata exactly — see chunks JSONL.
    # Key examples: 'Akbank', 'GarantiBBVA', 'NuhCimento', 'AdanaCimento', etc.
    queries = [
        ("Akbank'ın karbon emisyon azaltma hedefleri nelerdir?", "Akbank", None),
        ("Garanti BBVA'nın toplam geliri ne kadar?", "GarantiBBVA", None),
        ("Çimento şirketlerinin sürdürülebilirlik stratejileri", None, None),
    ]

    print("=" * 70)
    print("  SMOKE TEST — LLM ile uçtan uca 3 sorgu")
    print("=" * 70)

    t0 = time.time()
    llm = get_engine("transformers")
    print(f"[LLM load: {time.time()-t0:.1f}s]")

    rag = ReliabilityRAG(top_k=20, llm=llm)

    # Warm up: load embedder + NLI
    t0 = time.time()
    _ = rag.searcher  # triggers load
    _ = rag.nli       # triggers load
    print(f"[Embedder+NLI warm: {time.time()-t0:.1f}s]")

    print("\n" + "=" * 70)
    print("  WARMUP QUERY (discard timings)")
    print("=" * 70)
    _ = rag.query(
        "test", top_k=5, use_nli_filter=False, use_isolated_answering=False
    )

    print("\n" + "=" * 70)
    print("  REAL QUERIES")
    print("=" * 70)

    for i, (q, company, year) in enumerate(queries, 1):
        print(f"\n{'-' * 70}")
        print(f"  SORU {i}: {q}")
        if company:
            print(f"  Filtre: sirket={company}")
        print(f"{'-' * 70}")

        t0 = time.time()
        result = rag.query(
            q,
            top_k=20,
            company_filter=company,
            year_filter=year,
            use_nli_filter=True,
            use_isolated_answering=False,
        )
        total = time.time() - t0

        print(f"\n  📊 Timings:")
        for stage, t in result["timings"].items():
            print(f"     {stage:22s} {t:6.1f}s")
        print(f"     {'TOTAL':22s} {total:6.1f}s")

        print(f"\n  📈 Pipeline:")
        print(f"     Retrieved:   {result['retrieved_chunks']}")
        print(f"     Kept (clean): {result['filtered_chunks']}")
        print(f"     Removed:     {result['removed_chunks']}")
        print(f"     Contradiction edges: {result['contradiction_edges']}")

        print(f"\n  💬 ANSWER:")
        for line in result["answer"].split("\n"):
            print(f"     {line}")

        print(f"\n  📚 Top 3 sources:")
        for s in result["chunks_detail"][:3]:
            print(
                f"     - [{s['company']} {s['year']}] "
                f"({s['section_type']}, sim={s['score']:.3f})"
            )

    print(f"\n{'=' * 70}")
    print(f"  SMOKE TEST DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
