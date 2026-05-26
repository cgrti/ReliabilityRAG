"""
Tam ReliabilityRAG Pipeline Demo

Retrieval → Isolated Answering → NLI → MWIS Filtreleme → Final Cevap

Kullanım:
    python demo_full_pipeline.py                  # LLM'siz (stub mode)
    python demo_full_pipeline.py --llm            # Lokal LLM (Phi-3.5-mini)
    python demo_full_pipeline.py --llm --backend=ollama  # Ollama backend
"""
import sys
from rag_pipeline import ReliabilityRAG
from llm_engine import get_engine


def main():
    use_llm = "--llm" in sys.argv
    llm = None

    print("=" * 70)
    print("  ReliabilityRAG — Tam Pipeline Demo")
    print("  Retrieval → NLI → Çelişki Grafı → MWIS Filtreleme → Cevap")
    print("=" * 70)

    if use_llm:
        backend = "transformers"
        for arg in sys.argv:
            if arg.startswith("--backend="):
                backend = arg.split("=")[1]
        print(f"\n  LLM backend: {backend}")
        llm = get_engine(backend)
    else:
        print("\n  [Stub mode — LLM yok, ham kaynaklar gösterilir]")
        print("  LLM ile çalıştırmak için: python demo_full_pipeline.py --llm")

    rag = ReliabilityRAG(top_k=20, llm=llm)
    print("\nSistem hazır! Sorunuzu yazın. Çıkmak için boş bırakın.")
    print("Filtre: @ŞirketAdı @Yıl (ör: karbon hedefleri @Akbank @2023)\n")

    while True:
        query = input("Soru: ").strip()
        if not query:
            break

        # Parse optional filters
        company = None
        year = None
        words = query.split()
        clean_words = []
        for w in words:
            if w.startswith("@") and w[1:].isdigit():
                year = int(w[1:])
            elif w.startswith("@"):
                company = w[1:]
            else:
                clean_words.append(w)
        clean_query = " ".join(clean_words)

        if company:
            print(f"  [Filtre: şirket={company}]")
        if year:
            print(f"  [Filtre: yıl={year}]")

        print("\n  Pipeline çalışıyor...\n")
        result = rag.query(
            clean_query,
            company_filter=company,
            year_filter=year,
            use_nli_filter=True,
        )

        # Pipeline stats
        print(f"  ┌─ Pipeline Sonuçları ─────────────────────────")
        print(f"  │ Getirilen chunk:   {result['retrieved_chunks']}")
        print(f"  │ Filtrelenen chunk: {result['filtered_chunks']}")
        print(f"  │ Elenen chunk:      {result['removed_chunks']}")
        print(f"  │ Çelişki kenarı:    {result['contradiction_edges']}")
        print(f"  │ Toplam süre:       {result['total_time']:.2f}s")
        for stage, t in result['timings'].items():
            print(f"  │   {stage}: {t:.2f}s")
        print(f"  └──────────────────────────────────────────────")

        # Show answer
        print(f"\n  CEVAP:")
        print(f"  {'─'*50}")
        for line in result['answer'].split('\n'):
            print(f"  {line}")
        print(f"  {'─'*50}")

        # Show top 5 sources
        print(f"\n  Kaynaklar (ilk 5):")
        for i, d in enumerate(result['chunks_detail'][:5]):
            print(f"  {i+1}. [{d['company']} {d['year']}] "
                  f"({d['section_type']}, güv={d['reliability']}, "
                  f"sim={d['score']:.3f})")
        print()

    print("Görüşürüz!")


if __name__ == "__main__":
    main()
