"""
İnteraktif Arama Demo — Raporlarda soru sor!

Kullanım:
    python demo_search.py           # Genel arama
    python demo_search.py Akbank    # Sadece Akbank raporlarında ara

İstediğin soruyu yaz, en alakalı 10 rapor parçasını gösterir.
Çıkmak için boş bırak.
"""
import sys
from embedder import NumpyVectorSearch


def main():
    company_filter = sys.argv[1] if len(sys.argv) > 1 else None

    print("Vektör veritabanı yükleniyor...")
    searcher = NumpyVectorSearch()

    if company_filter:
        print(f"\nFiltre: sadece '{company_filter}' raporları")
    print("Sorgunuzu yazın. Çıkmak için boş bırakın.\n")

    while True:
        query = input("Soru: ").strip()
        if not query:
            break

        results = searcher.search(query, top_k=10, company=company_filter)

        if not results:
            print("  Sonuç bulunamadı.\n")
            continue

        print(f"\n  {len(results)} sonuç bulundu:\n")
        for i, r in enumerate(results):
            print(f"  {i+1}. [{r['company']} {r['year']}] "
                  f"({r['section_type']}, benzerlik={r['score']:.3f})")
            print(f"     Başlık: {r['heading'][:80]}")
            # Show first 200 chars of text, nicely wrapped
            text_preview = r['text'][:200].replace('\n', ' ')
            print(f"     {text_preview}...")
            print()


if __name__ == "__main__":
    main()
