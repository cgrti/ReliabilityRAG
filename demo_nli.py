"""
Hızlı NLI Demo — Kendi cümlelerini test et!

Kullanım:
    python demo_nli.py

İki cümle gir, model çelişki/uyum/nötr olduğunu söylesin.
Çıkmak için boş bırak.
"""
from nli_graph import NLIContradictionGraph

def main():
    print("NLI Çelişki Tespit Modeli yükleniyor...")
    nli = NLIContradictionGraph()
    print("\nModel hazır! İki cümle girin, çelişki olup olmadığını test edin.")
    print("Çıkmak için boş bırakın.\n")

    while True:
        print("-" * 60)
        s1 = input("Cümle 1: ").strip()
        if not s1:
            break
        s2 = input("Cümle 2: ").strip()
        if not s2:
            break

        result = nli.predict_nli_batch([s1], [s2])[0]
        best = max(result, key=result.get)

        labels_tr = {
            "entailment": "UYUMLU (aynı şeyi söylüyor)",
            "neutral": "NÖTR (ilişkisiz/bağımsız)",
            "contradiction": "ÇELİŞKİLİ!",
        }

        print(f"\n  Sonuç: {labels_tr.get(best, best)}")
        print(f"  Detay: çelişki={result.get('contradiction',0):.2%}, "
              f"uyum={result.get('entailment',0):.2%}, "
              f"nötr={result.get('neutral',0):.2%}")
        print()


if __name__ == "__main__":
    main()
