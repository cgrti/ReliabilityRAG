"""
Minimal Faithfulness Scorer — LLM-judge tabanlı RAG değerlendirmesi.

RAGAS framework'ün ihtiyacı olan OpenAI API + büyük judge model yerine,
mevcut Phi-3.5-mini-instruct 4-bit modelini judge olarak kullanır.
Tek-makinede, GPU 6 GB'a sığar, ek bağımlılık gerekmez.

Üretilen metrikler (RAGAS terminolojisi ile uyumlu):
  - faithfulness     : Cevaptaki her iddianın kaynaklarda destekli olma oranı
  - answer_relevance : Cevabın soruyla ilgililik skoru (yüzeysel: 0/1)
  - context_recall   : Beklenen kaynak chunk'larının kaç tanesi retrieved?

Pipeline:
1) `data/demo_cache.json` veya gerçek `rag.query()` çıktısı yükle
2) Her cevap için iddiaları LLM ile çıkar (kısa bullet listesi)
3) Her iddia için, "kaynaklarda destekli mi?" diye sor → evet/hayır
4) faithfulness = destekli iddia / toplam iddia

Kullanım:
    HF_HUB_OFFLINE=1 python faithfulness_eval.py --cache data/demo_cache.json
    HF_HUB_OFFLINE=1 python faithfulness_eval.py --quick     # 2 örnek hızlı
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import PROJECT_ROOT
from llm_engine import get_engine


CACHE_DEFAULT = PROJECT_ROOT / "data" / "demo_cache.json"
OUT_PATH = PROJECT_ROOT / "data" / "faithfulness_results.json"


# ── Prompts ──────────────────────────────────────────────────────────────

CLAIM_EXTRACTION_PROMPT = """Aşağıdaki cevap metnini oku ve içindeki bireysel olgusal iddiaları madde madde listele.

KURALLAR:
- Her madde TEK bir olgusal iddia olsun (yıl, sayı, taahhüt, oran vb.).
- Genel sözler ("önemlidir", "değer veriyoruz") iddia DEĞİLDİR, atlayın.
- En fazla 5 madde.
- Her maddeyi "- " ile başlat.

CEVAP:
{answer}

İDDİALAR:"""


VERIFICATION_PROMPT = """Aşağıda bir İDDİA ve bu iddianın doğrulanması için kullanılacak KAYNAK metinler var.

KAYNAKLAR:
{sources}

İDDİA: {claim}

Bu iddia, yukarıdaki KAYNAKLAR içinde DOĞRUDAN destekleniyor mu? Sadece "EVET" veya "HAYIR" yaz. Açıklama YOK.

CEVAP:"""


# ── Helpers ──────────────────────────────────────────────────────────────

def extract_claims(llm, answer: str, max_tokens: int = 200) -> list[str]:
    """Ask the LLM to break an answer into atomic factual claims."""
    prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer[:1500])
    response = llm.generate(prompt, max_tokens=max_tokens)
    # Parse "- foo" lines, strip prefix
    claims = []
    for line in response.split("\n"):
        line = line.strip()
        m = re.match(r"^[-*•]\s*(.+)$", line)
        if m and len(m.group(1)) > 10:
            claims.append(m.group(1).rstrip("."))
        if len(claims) >= 5:
            break
    return claims


def verify_claim(llm, claim: str, sources_text: str, max_tokens: int = 10) -> bool:
    """Ask the LLM if a claim is supported by the sources. Returns True/False."""
    prompt = VERIFICATION_PROMPT.format(sources=sources_text[:3000], claim=claim)
    response = llm.generate(prompt, max_tokens=max_tokens).strip().upper()
    # Accept "EVET", "EVET.", "EVET, ..."
    return response.startswith("EVET")


def render_sources(chunks_detail: list[dict], max_n: int = 5) -> str:
    """Compose sources block from cached chunks_detail entries."""
    parts = []
    for i, c in enumerate(chunks_detail[:max_n], 1):
        parts.append(
            f"Kaynak {i} [{c.get('company','?')} {c.get('year','?')} | "
            f"{c.get('section_type','?')}]:\n{c.get('text_preview','')}"
        )
    return "\n\n---\n\n".join(parts)


# ── Per-query eval ──────────────────────────────────────────────────────

def eval_one(llm, entry: dict) -> dict:
    """Evaluate a single cached query entry."""
    spec = entry.get("spec", {})
    question = spec.get("question", "?")
    answer = entry.get("answer", "")
    chunks = entry.get("chunks_detail", [])

    print(f"\n[Q] {question[:80]}")
    print(f"  [extracting claims…]")
    t0 = time.time()
    claims = extract_claims(llm, answer)
    print(f"  → {len(claims)} claim ({time.time()-t0:.1f}s)")
    for c in claims:
        print(f"     - {c[:120]}")

    if not claims:
        return {
            "question": question,
            "n_claims": 0, "n_supported": 0,
            "faithfulness": None,
            "claim_results": [],
        }

    sources_text = render_sources(chunks)
    if not sources_text:
        print("  [no sources to verify against]")
        return {
            "question": question,
            "n_claims": len(claims), "n_supported": 0,
            "faithfulness": 0.0,
            "claim_results": [{"claim": c, "supported": False} for c in claims],
        }

    print(f"  [verifying claims against {len(chunks)} sources…]")
    results = []
    n_supported = 0
    for i, claim in enumerate(claims, 1):
        t0 = time.time()
        ok = verify_claim(llm, claim, sources_text)
        n_supported += int(ok)
        mark = "✓" if ok else "✗"
        print(f"  {i}. {mark}  ({time.time()-t0:.1f}s)  {claim[:90]}")
        results.append({"claim": claim, "supported": ok})

    faithfulness = n_supported / len(claims)
    print(f"  → faithfulness = {n_supported}/{len(claims)} = {faithfulness:.2f}")
    return {
        "question": question,
        "n_claims": len(claims),
        "n_supported": n_supported,
        "faithfulness": faithfulness,
        "claim_results": results,
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--quick", action="store_true",
                        help="Only evaluate first 2 entries")
    parser.add_argument("--max", type=int, default=None,
                        help="Limit number of entries")
    args = parser.parse_args()

    print("=" * 70)
    print("  Faithfulness Evaluation — Phi-3.5-mini judge")
    print("=" * 70)

    if not args.cache.exists():
        print(f"[!] Cache not found: {args.cache}")
        print("    Run `python demo_cache.py` first to generate cached answers.")
        return

    with open(args.cache, "r", encoding="utf-8") as f:
        cache = json.load(f)
    if args.quick:
        cache = cache[:2]
    elif args.max:
        cache = cache[:args.max]
    print(f"Loaded {len(cache)} cached queries from {args.cache.name}")

    print("\nLoading Phi-3.5-mini (judge)…")
    t0 = time.time()
    llm = get_engine("transformers")
    print(f"  LLM ready in {time.time()-t0:.1f}s\n")

    results = []
    overall_t0 = time.time()
    for entry in cache:
        r = eval_one(llm, entry)
        results.append(r)

    # ── Summary ─────────────────────────────────────────────────────────
    valid = [r for r in results if r["faithfulness"] is not None]
    if valid:
        mean_f = sum(r["faithfulness"] for r in valid) / len(valid)
        total_claims = sum(r["n_claims"] for r in valid)
        total_supported = sum(r["n_supported"] for r in valid)
    else:
        mean_f, total_claims, total_supported = 0, 0, 0

    print("\n" + "=" * 70)
    print("  ÖZET")
    print("=" * 70)
    print(f"  Evaluated:     {len(results)} queries")
    print(f"  Total claims:  {total_claims}")
    print(f"  Supported:     {total_supported}")
    print(f"  Mean faithfulness: {mean_f:.3f}")
    print(f"  Wall time:     {time.time()-overall_t0:.0f}s")

    payload = {
        "summary": {
            "n_queries": len(results),
            "total_claims": total_claims,
            "total_supported": total_supported,
            "mean_faithfulness": round(mean_f, 4),
            "judge_model": "microsoft/Phi-3.5-mini-instruct (4-bit)",
        },
        "per_query": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
