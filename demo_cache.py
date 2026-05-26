"""
Demo Cache Builder — Pre-compute Gradio demo queries.

SUNUM ÖNCESİ KOŞ (tek seferlik, ~5-10 dk):
    python demo_cache.py

Çıktı: data/demo_cache.json — Gradio app'in "cached" toggle'u bunu okur.

NEDEN:  Canlı demoda LLM 2-3 dk sürer. Pre-compute ile anında sonuç.
        "Öncesinde hesapladım, arkada gerçekten bu çalışıyor" dersin.

Note: LLM non-deterministic (temperature=0.3, do_sample=True), yani her koşum
      farklı cevap verir. Cache önceki koşumun cevabını dondurur.
"""
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from rag_pipeline import ReliabilityRAG
from llm_engine import get_engine


DEMO_QUERIES = [
    {
        "question": "Akbank'ın karbon emisyon azaltma hedefleri nelerdir?",
        "company": "Akbank",
        "year": None,
    },
    {
        "question": "Garanti BBVA'nın sürdürülebilirlik performansı ve gelir yapısı nasıl?",
        "company": "GarantiBBVA",
        "year": None,
    },
    {
        "question": "Nuh Çimento'nun enerji verimliliği ve emisyon yönetimi stratejisi",
        "company": "NuhCimento",
        "year": None,
    },
    {
        "question": "Kadın istihdamı ve fırsat eşitliği politikaları",
        "company": None,
        "year": None,
    },
    {
        "question": "Yenilenebilir enerji yatırımları",
        "company": None,
        "year": 2023,
    },
]


def main():
    out_path = Path(__file__).parent / "data" / "demo_cache.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DEMO CACHE BUILDER")
    print("=" * 70)
    print(f"Target: {out_path}")
    print(f"Queries: {len(DEMO_QUERIES)}")
    print()

    t0 = time.time()
    llm = get_engine("transformers")
    print(f"[LLM load: {time.time()-t0:.1f}s]\n")

    rag = ReliabilityRAG(top_k=20, llm=llm)
    t0 = time.time()
    _ = rag.searcher
    _ = rag.nli
    print(f"[Embedder+NLI warm: {time.time()-t0:.1f}s]\n")

    # Warmup (discard)
    print("[warmup query...]")
    _ = rag.query("test", top_k=5, use_nli_filter=False, use_isolated_answering=False)
    print("[warmup done]\n")

    cache = []
    for i, spec in enumerate(DEMO_QUERIES, 1):
        print("=" * 70)
        print(f"  [{i}/{len(DEMO_QUERIES)}] {spec['question'][:60]}")
        if spec["company"]:
            print(f"  Filter: {spec['company']}")
        if spec["year"]:
            print(f"  Year: {spec['year']}")
        print("=" * 70)

        t0 = time.time()
        result = rag.query(
            spec["question"],
            top_k=20,
            company_filter=spec["company"],
            year_filter=spec["year"],
            use_nli_filter=True,
            use_isolated_answering=False,
        )
        elapsed = time.time() - t0

        # Trim result to cacheable form (drop large fields, keep UI-friendly)
        cached = {
            "spec": spec,
            "answer": result["answer"],
            "retrieved_chunks": result["retrieved_chunks"],
            "filtered_chunks": result["filtered_chunks"],
            "removed_chunks": result["removed_chunks"],
            "contradiction_edges": result["contradiction_edges"],
            "timings": result["timings"],
            "total_time": result["total_time"],
            "chunks_detail": result["chunks_detail"][:10],  # cap at 10
            "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_seconds": round(elapsed, 2),
        }
        cache.append(cached)

        print(f"\n  Time: {elapsed:.1f}s")
        print(f"  Retrieved: {cached['retrieved_chunks']} | "
              f"Kept: {cached['filtered_chunks']} | "
              f"Edges: {cached['contradiction_edges']}")
        print(f"  Answer preview: {cached['answer'][:150]}...\n")

        # Save incrementally so partial failure doesn't lose everything
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"  CACHE WRITTEN: {out_path}")
    print(f"  Total queries: {len(cache)}")
    print(f"  Total wall time: {sum(c['wall_seconds'] for c in cache):.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
