"""
make_demo_cache.py — Builds data/demo_cache.json by running curated demo queries
through the full RAG pipeline. cached_demo.py reads this as a bullet-proof
backup demo (no GPU needed at presentation time — answers pre-computed).

Each entry captures: spec (question + optional filters), final answer, pipeline
metrics, chunks_detail (source previews), and wall-clock timings — everything
needed for cached_demo.py to render a faithful replay.

Run on Akvaryum (needs LLM):
    python make_demo_cache.py                  # full run, ~5-10 minutes
    python make_demo_cache.py --limit 3        # quick smoke test
    python make_demo_cache.py --append         # add to existing cache

Curate queries below: pick a balance of (a) single-company simple, (b) multi-
company synthesis, (c) sector-wide, (d) temporal evolution, (e) numerical fact,
(f) greenwashing-prone (interdepartmental contradiction). Aim 10-15 entries.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from config import PROJECT_ROOT

CACHE_PATH = PROJECT_ROOT / "data" / "demo_cache.json"


# ── Curated demo queries ────────────────────────────────────────────────
# Companies & topics confirmed present in the 248-report corpus.
# `showcase` is metadata — explains what this query demonstrates to a viewer.

DEMO_QUERIES = [
    # 1. Single-company environment — basic RAG showcase
    {
        "question": "Akbank'ın karbon emisyon azaltma hedefleri nelerdir?",
        "company": None, "year": None,
        "showcase": "Single-company environmental query — basic retrieval+synthesis",
    },
    # 2. Single-company financial — numerical retrieval
    {
        "question": "Garanti BBVA'nın toplam aktif büyüklüğü ve kredi portföyü ne kadardır?",
        "company": None, "year": None,
        "showcase": "Financial numerical extraction from source documents",
    },
    # 3. Sector comparison — cross-document synthesis
    {
        "question": "Çimento sektöründe alternatif yakıt kullanım oranı en yüksek hangi şirkettedir?",
        "company": None, "year": None,
        "showcase": "Sector-wide cross-document comparison",
    },
    # 4. Multi-company social — proven Turkish-Llama case (2026-05-30 test)
    {
        "question": "Kadın istihdamı ve fırsat eşitliği politikaları hakkında bilgi ver",
        "company": None, "year": None,
        "showcase": "Multi-company social policy synthesis (production-proven query)",
    },
    # 5. Single-company multi-section synthesis
    {
        "question": "Anadolu Hayat Emeklilik'in sürdürülebilirlik politikaları ve hedefleri nelerdir?",
        "company": None, "year": None,
        "showcase": "Single-company across env/social/governance sections",
    },
    # 6. Temporal multi-year evolution
    {
        "question": "Türk Telekom'un dijital dönüşüm yatırımları son yıllarda nasıl değişti?",
        "company": None, "year": None,
        "showcase": "Temporal evolution across multiple years",
    },
    # 7. Greenwashing — interdepartmental contradiction check
    {
        "question": "Tüpraş'ın yenilenebilir enerji yatırımları ile fosil yakıt bağımlılığı arasındaki ilişki nedir?",
        "company": None, "year": None,
        "showcase": "Greenwashing-prone query — contradiction filter showcase",
    },
    # 8. Sector benchmark
    {
        "question": "Bankacılık sektöründe kadın yönetici oranı nasıl değişmektedir?",
        "company": None, "year": None,
        "showcase": "Sector-wide benchmark with temporal angle",
    },
    # 9. Single-company ESG overview
    {
        "question": "Vakıf GYO'nun çevre, sosyal ve yönetim (ESG) politikaları nelerdir?",
        "company": None, "year": None,
        "showcase": "Single-company comprehensive ESG view",
    },
    # 10. Filtered query — company + year
    {
        "question": "Akbank 2023 sürdürülebilirlik raporundaki ana başlıklar nelerdir?",
        "company": "Akbank", "year": 2023,
        "showcase": "Filtered retrieval (company + year metadata filter)",
    },
    # 11. Precise numerical
    {
        "question": "Garanti BBVA'nın Scope 1 ve Scope 2 sera gazı emisyonları kaç ton CO2e seviyesindedir?",
        "company": None, "year": None,
        "showcase": "Scope-disambiguated numerical query (NLI scope-difference test)",
    },
    # 12. Broad cross-sector
    {
        "question": "Türk şirketlerinin sürdürülebilirlik raporlarında öne çıkan ortak temalar nelerdir?",
        "company": None, "year": None,
        "showcase": "Broad cross-sector thematic synthesis",
    },
    # ─ Round 2 (added 2026-05-30) — 5 additional queries for richer demo cache ─
    # All companies below were confirmed in the corpus via prior demo_cache outputs.
    # 13. Single-co governance (Yapı Kredi appeared in bankacılık query)
    {
        "question": "Yapı Kredi'nin kurumsal yönetim komiteleri ve denetim mekanizmaları nelerdir?",
        "company": None, "year": None,
        "showcase": "Single-company governance — organizational structure synthesis",
    },
    # 14. Single-co env + temporal (Anadolu Efes appeared in cross-sector query)
    {
        "question": "Anadolu Efes'in su tüketimi ve geri dönüşüm uygulamaları nasıl ilerlemiştir?",
        "company": None, "year": None,
        "showcase": "Environmental performance with temporal evolution",
    },
    # 15. Financial product (TSKB appeared in bankacılık query)
    {
        "question": "TSKB'nin sürdürülebilir finansman ürünleri ve yeşil tahvil yatırımları nelerdir?",
        "company": None, "year": None,
        "showcase": "Financial product portfolio — sustainability-linked instruments",
    },
    # 16. Sector ESG cross-cut
    {
        "question": "Türk bankacılık sektöründe ESG risk değerlendirme yaklaşımları arasındaki farklar nelerdir?",
        "company": None, "year": None,
        "showcase": "Sector-wide ESG methodology comparison",
    },
    # 17. Single-co social numerical (Borusan appeared in kadın istihdamı query)
    {
        "question": "Borusan Holding'in iş sağlığı ve güvenliği performansı ve kaza oranları nasıl gelişmiştir?",
        "company": None, "year": None,
        "showcase": "İSG numerical reporting with temporal angle",
    },
]


def build_cache_entry(rag, query_spec: dict) -> dict:
    """Run one query through the full pipeline, wrap in cache format."""
    print(f"  → {query_spec['question'][:75]}")
    print(f"    showcase: {query_spec.get('showcase', '')[:90]}")

    t_wall = time.time()
    result = rag.query(
        question=query_spec["question"],
        company_filter=query_spec.get("company"),
        year_filter=query_spec.get("year"),
    )
    wall = time.time() - t_wall

    print(f"    kept {result['filtered_chunks']}/{result['retrieved_chunks']}, "
          f"{result['contradiction_edges']} edges, "
          f"final_gen {result['timings'].get('final_generation', 0):.1f}s, "
          f"TOTAL {result['total_time']:.1f}s")

    return {
        "spec": {
            "question": query_spec["question"],
            "company": query_spec.get("company"),
            "year": query_spec.get("year"),
            "showcase": query_spec.get("showcase", ""),
        },
        "answer": result["answer"],
        "retrieved_chunks": result["retrieved_chunks"],
        "filtered_chunks": result["filtered_chunks"],
        "removed_chunks": result["removed_chunks"],
        "contradiction_edges": result["contradiction_edges"],
        "timings": result["timings"],
        "total_time": result["total_time"],
        "chunks_detail": result["chunks_detail"],
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Build demo_cache.json from curated queries")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N queries (debug)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing cache instead of overwriting")
    parser.add_argument("--out", type=Path, default=CACHE_PATH,
                        help=f"Output path (default: {CACHE_PATH})")
    parser.add_argument("--backend", default="transformers",
                        help="LLM backend (transformers|ollama|api)")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    queries = DEMO_QUERIES[: args.limit] if args.limit else DEMO_QUERIES
    print(f"Building demo cache: {len(queries)} queries → {args.out}")
    print(f"(Estimated time: ~{len(queries) * 30}s = {len(queries) * 30 // 60} min)")
    print()

    # Load existing if --append
    existing = []
    if args.append and args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Appending to {len(existing)} existing entries\n")

    # Init RAG with LLM
    print("Loading LLM + RAG pipeline...")
    from rag_pipeline import ReliabilityRAG
    from llm_engine import get_engine
    t0 = time.time()
    llm = get_engine(args.backend)
    rag = ReliabilityRAG(top_k=20, llm=llm)
    print(f"  ready in {time.time() - t0:.1f}s\n")

    # Build entries
    entries = list(existing)
    failures = []
    for i, qs in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}]")
        try:
            entry = build_cache_entry(rag, qs)
            entries.append(entry)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            failures.append((i, qs["question"], str(e)))
        print()

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"{'='*70}")
    print(f"  Saved {len(entries)} entries to {args.out}")
    if failures:
        print(f"  ⚠ {len(failures)} failures:")
        for i, q, e in failures:
            print(f"    [{i}] {q[:60]}... — {e[:60]}")
    print(f"{'='*70}")
    print()
    print("Sonraki adım:")
    print(f"  python cached_demo.py   # backup demo viewer, port 7861")


if __name__ == "__main__":
    main()
