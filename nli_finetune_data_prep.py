"""
NLI Fine-tuning Veri Hazırlığı — Aday Çelişki Pair'leri Çıkarma

NIYE: Mevcut mDeBERTa-v3-base Türkçe revize-tipi cümleleri yakalayamıyor
("2030 hedefi %50" vs "%50'den %35'e revize edilmiştir" → NLI neutral diyor).
5080'de mDeBERTa-large'i Türkçe çelişki örnekleri üzerinde fine-tune etmek için
**etiketli veri lazım**: 100+ pair, her biri contradiction/entailment/neutral.

Bu script, 248 rapor chunk'ından otomatik **aday pair'ler** çıkarır:
  1) Bir topic listesi al (örn. karbon hedefi, kadın istihdamı, enerji yatırımı)
  2) Her topic için her şirketin farklı yıllardaki chunk'larını NumpyVectorSearch
     ile getir (aynı şirket, farklı yıl → temporal kontrast adayı)
  3) Pair'lerden NLI'yi geçir, contradiction prob ≥ 0.3 olanları kaydet
  4) JSONL çıktı: data/nli_finetune_candidates.jsonl
     Her satır: {premise, hypothesis, model_pred, model_prob, ground_truth_label,
                  company_a, year_a, company_b, year_b, topic}
  5) Kullanıcı `ground_truth_label` alanını manuel doldurur (entailment/contradiction/neutral)

Kullanım (5080'de):
    HF_HUB_OFFLINE=1 python nli_finetune_data_prep.py --target-pairs 150
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import PROJECT_ROOT
from embedder import NumpyVectorSearch
from nli_graph import NLIContradictionGraph


OUT_PATH = PROJECT_ROOT / "data" / "nli_finetune_candidates.jsonl"


# Topics that historically yield rich temporal/scope conflicts.
TOPICS = [
    "karbon emisyon azaltma hedefi",
    "yenilenebilir enerji kullanım oranı",
    "kadın çalışan ve yönetici oranı",
    "iş sağlığı güvenliği kaza istatistikleri",
    "atık geri dönüşüm oranı",
    "enerji tüketimi GWh",
    "su tüketimi azaltımı",
    "toplam çalışan sayısı",
    "net kar zarar finansal sonuç",
    "ar-ge harcaması yatırımı",
    "kredi portföyü karbon yoğunluğu",
    "tedarikçi sürdürülebilirlik denetimi",
]


def collect_chunks_per_topic(searcher: NumpyVectorSearch,
                             topic: str, per_company: int = 4) -> dict:
    """
    Bir topic için her şirketten en alakalı `per_company` chunk getir.
    Şirket bazında grupla ki sonra **aynı şirket × farklı yıl** pair'leri kurabilelim.
    """
    # Geniş çekiş yap, sonra şirket başına filtrele
    raw = searcher.search(topic, top_k=400)
    by_company: dict[str, list[dict]] = {}
    for r in raw:
        bucket = by_company.setdefault(r["company"], [])
        if len(bucket) < per_company:
            bucket.append(r)
    return by_company


def build_pairs(by_company: dict, topic: str) -> list[dict]:
    """
    Aynı şirket içindeki farklı-yıl chunk çiftleri = temporal kontrast adayları.
    Buna ek olarak: ÇOK az şirket sayısında, farklı şirketler arası aynı yıl pair'leri
    (cross-company conflict adayı).
    """
    pairs = []
    # 1) Same-company × different-year
    for company, chunks in by_company.items():
        for a, b in itertools.combinations(chunks, 2):
            if a["year"] == b["year"]:
                continue  # same year same company → düşük öncelik
            pairs.append({
                "kind": "same_company_different_year",
                "topic": topic,
                "premise":   a["text"][:600],
                "hypothesis": b["text"][:600],
                "company_a": a["company"], "year_a": a["year"], "sec_a": a["section_type"],
                "company_b": b["company"], "year_b": b["year"], "sec_b": b["section_type"],
            })
    # 2) Cross-company × same-year (sample only first 10 to limit volume)
    companies = list(by_company.keys())
    cross = 0
    for ca, cb in itertools.combinations(companies, 2):
        for a in by_company[ca]:
            for b in by_company[cb]:
                if a["year"] != b["year"]:
                    continue
                if cross >= 10:
                    break
                pairs.append({
                    "kind": "cross_company_same_year",
                    "topic": topic,
                    "premise":   a["text"][:600],
                    "hypothesis": b["text"][:600],
                    "company_a": a["company"], "year_a": a["year"], "sec_a": a["section_type"],
                    "company_b": b["company"], "year_b": b["year"], "sec_b": b["section_type"],
                })
                cross += 1
    return pairs


def score_with_nli(nli: NLIContradictionGraph, pairs: list[dict],
                   min_contradiction: float = 0.30) -> list[dict]:
    """Pair'leri NLI'ye sok, contradiction prob ≥ min olanları döndür (sırasız)."""
    if not pairs:
        return []
    premises = [p["premise"] for p in pairs]
    hypotheses = [p["hypothesis"] for p in pairs]
    results = nli.predict_nli_batch(premises, hypotheses)
    kept = []
    for p, r in zip(pairs, results):
        c = r.get("contradiction", 0.0)
        e = r.get("entailment", 0.0)
        n = r.get("neutral", 0.0)
        p["model_prob"] = {"contradiction": round(c, 3),
                           "entailment": round(e, 3),
                           "neutral": round(n, 3)}
        p["model_pred"] = max(p["model_prob"], key=p["model_prob"].get)
        # User fills this in manually
        p["ground_truth_label"] = ""
        if c >= min_contradiction:
            kept.append(p)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-pairs", type=int, default=150,
                        help="Target number of candidate pairs to emit")
    parser.add_argument("--per-company", type=int, default=4,
                        help="Chunks per company per topic")
    parser.add_argument("--min-contradiction", type=float, default=0.30,
                        help="Filter cutoff on NLI contradiction probability")
    args = parser.parse_args()

    print("=" * 70)
    print("  NLI Fine-tuning Veri Hazırlığı — Aday Pair Çıkarımı")
    print("=" * 70)

    print("\n[1/3] Loading searcher + NLI...")
    t0 = time.time()
    s = NumpyVectorSearch()
    nli = NLIContradictionGraph()
    print(f"      Ready in {time.time()-t0:.1f}s")

    print(f"\n[2/3] Generating candidate pairs across {len(TOPICS)} topics...")
    all_candidates = []
    for topic in TOPICS:
        if len(all_candidates) >= args.target_pairs:
            break
        t0 = time.time()
        by_company = collect_chunks_per_topic(s, topic, per_company=args.per_company)
        pairs = build_pairs(by_company, topic)
        kept = score_with_nli(nli, pairs, args.min_contradiction)
        print(f"  · {topic[:35]:35s}  {len(pairs):>4} pair → {len(kept):>3} kept "
              f"({time.time()-t0:.1f}s)")
        all_candidates.extend(kept)

    # Sort by contradiction prob descending so top adays are first
    all_candidates.sort(key=lambda p: -p["model_prob"]["contradiction"])
    all_candidates = all_candidates[:args.target_pairs]

    print(f"\n[3/3] Writing {len(all_candidates)} pairs to {OUT_PATH}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p in all_candidates:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nDone. Next step:")
    print(f"  1) Open {OUT_PATH.name} in a text editor")
    print(f"  2) For each pair, fill `ground_truth_label`: ")
    print(f"     - \"contradiction\" if claims directly conflict")
    print(f"     - \"entailment\"    if one implies the other")
    print(f"     - \"neutral\"       if unrelated / no relation")
    print(f"  3) Save and feed into mDeBERTa fine-tuning script (TODO).")


if __name__ == "__main__":
    main()
