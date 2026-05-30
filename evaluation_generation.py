"""
evaluation_generation.py — LLM Output Faithfulness Eval (Stage B, 2026-05-30)

Complementary to evaluation.py (filter-layer eval). This measures the
LLM-layer of the reliability claim: does the final answer faithfully
preserve the entities, years, and numbers present in the source chunks,
or does the LLM hallucinate / drift?

Motivation: Turkish-Llama-8b production test (2026-05-30) showed entity
drift in summary ("Anadolu Hayat Emeklilik" → "Anadolu Hayat Sigortacılığı"),
acronym hallucination ("FEM" → "FEZ"), even after the chat-template fix.
The filter-layer (NLI+GAT) is doing its job; the LLM layer needs its own
quantitative reliability measurement for the thesis.

Metrics:
- entity_recall:       sources'taki şirketler kaç tanesi çıktıda?
- entity_purity:       çıktıdaki şirketlerin hepsi kaynaklarda var mı?
- year_accuracy:       çıktıdaki yıllar kaynak yıllarıyla eşleşiyor mu?
- numerical_fidelity:  çıktıdaki sayılar kaynak sayılarıyla ±5% eşleşiyor mu?

Modes:
- oracle:    feed only expected_kept chunks (measures LLM alone)
- pipeline:  feed all chunks through actual NLI+GAT filter, then LLM
             (measures end-to-end pipeline including filter mistakes)

Usage (Akvaryum, needs LLM):
  python evaluation_generation.py --mode oracle
  python evaluation_generation.py --mode pipeline
  python evaluation_generation.py --mode oracle --limit 5   # quick debug
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from config import PROJECT_ROOT
from rag_pipeline import ReliabilityRAG, RetrievedChunk
from synthetic_testset import TESTSET_PATH

REPORT_PATH = PROJECT_ROOT / "data" / "generation_eval_report.json"


# ── Number / Year extraction ───────────────────────────────────────────

# Turkish-aware number patterns. Order matters: longer/specific first.
_NUMBER_PATTERNS = [
    re.compile(r"%\s*(\d+(?:[.,]\d+)?)"),                              # %42, %42.5
    re.compile(r"\b(\d{1,3}(?:\.\d{3})+(?:[.,]\d+)?)\b"),             # 1.250, 245.000
    re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:milyon|milyar|bin)"),       # 245 milyon, 8.2 milyar
    re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:ton|GWh|kWh|MWh|m³|km)"),   # 45 ton, 290 GWh
    re.compile(r"\b(\d+(?:[.,]\d+)?)\b"),                              # bare numbers fallback
]
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def extract_numbers(text: str) -> set[float]:
    """
    Extract numeric values from text. Handles Turkish formatting
    (thousand separator '.', decimal separator ','). Excludes 4-digit
    years (1900-2099) since those are handled separately.
    """
    nums = set()
    for pat in _NUMBER_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(1)
            normalized = raw.replace(".", "").replace(",", ".")
            try:
                val = float(normalized)
            except ValueError:
                continue
            # Skip years (handled by extract_years)
            if 1900 <= val <= 2099 and val == int(val):
                continue
            nums.add(val)
    return nums


def extract_years(text: str) -> set[int]:
    return {int(m.group(1)) for m in _YEAR_RE.finditer(text)}


def extract_companies(text: str, candidates: set[str]) -> set[str]:
    """
    Substring search of known candidates in text (case-insensitive).
    Returns the subset of candidates that appear.
    """
    text_lower = text.lower()
    return {c for c in candidates if c.lower() in text_lower}


# ── Per-Test Scoring ───────────────────────────────────────────────────

@dataclass
class TestResult:
    test_id: str
    dimension: str
    question: str
    answer: str
    timings: dict = field(default_factory=dict)

    expected_companies: set = field(default_factory=set)
    mentioned_companies: set = field(default_factory=set)

    expected_years: set = field(default_factory=set)
    mentioned_years: set = field(default_factory=set)

    expected_numbers: set = field(default_factory=set)
    mentioned_numbers: set = field(default_factory=set)

    @property
    def entity_recall(self) -> float:
        if not self.expected_companies:
            return 1.0
        return len(self.mentioned_companies & self.expected_companies) / len(self.expected_companies)

    @property
    def entity_purity(self) -> float:
        """Of companies mentioned in output, fraction that exist in sources."""
        if not self.mentioned_companies:
            return 1.0
        return len(self.mentioned_companies & self.expected_companies) / len(self.mentioned_companies)

    @property
    def year_accuracy(self) -> float:
        """Of years mentioned in output, fraction that appear in sources."""
        if not self.mentioned_years:
            return 1.0
        return len(self.mentioned_years & self.expected_years) / len(self.mentioned_years)

    @property
    def numerical_fidelity(self) -> float:
        """Of numbers in output, fraction matching some source number (±5%)."""
        if not self.mentioned_numbers:
            return 1.0
        matched = 0
        for n in self.mentioned_numbers:
            for sn in self.expected_numbers:
                if sn == 0:
                    continue
                if abs(n - sn) / abs(sn) <= 0.05:  # 5% tolerance
                    matched += 1
                    break
        return matched / len(self.mentioned_numbers)


# ── Eval Loop ──────────────────────────────────────────────────────────

def _to_retrieved_chunks(test_id: str, raw_chunks: list[dict]) -> list[RetrievedChunk]:
    """Convert synthetic testset chunk dicts to RetrievedChunk objects."""
    return [
        RetrievedChunk(
            chunk_id=f"{test_id}-{i}",
            text=c["text"],
            company=c["company"],
            year=c["year"],
            section_type=c["section_type"],
            reliability_weight=c["reliability_weight"],
            heading="",
            page_start=0,
            score=1.0,
        )
        for i, c in enumerate(raw_chunks)
    ]


def evaluate_test(rag: ReliabilityRAG, test: dict, mode: str = "oracle") -> TestResult:
    """Run one test through the LLM and score the output."""
    all_raw = test["chunks"]

    if mode == "oracle":
        feed_indices = test["expected_kept"]
    elif mode == "pipeline":
        feed_indices = list(range(len(all_raw)))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    timings = {}
    feed_chunks = _to_retrieved_chunks(test["test_id"], [all_raw[i] for i in feed_indices])

    # Pipeline mode: run actual NLI+GAT filter on all chunks first
    if mode == "pipeline" and len(feed_chunks) > 1:
        t0 = time.time()
        kept_indices, _ = rag.nli.filter_chunks(
            [c.text for c in feed_chunks],
            [c.reliability_weight for c in feed_chunks],
            [c.year for c in feed_chunks],
            section_types=[c.section_type for c in feed_chunks],
            use_gat=True,
        )
        feed_chunks = [feed_chunks[i] for i in kept_indices]
        timings["nli_graph_filter"] = time.time() - t0

    # Generate final answer (max_sources=10 to cover dense_graph 8-chunk cases)
    t0 = time.time()
    answer = rag.final_answer(test["question"], feed_chunks, max_sources=10)
    timings["final_generation"] = time.time() - t0

    # Ground truth: everything we actually fed (oracle: expected_kept;
    # pipeline: whatever survived filter)
    fed_raw = [all_raw[i] for i in feed_indices]
    if mode == "pipeline":
        # Recompute fed_raw based on what survived filter
        survivor_ids = {c.chunk_id for c in feed_chunks}
        # rebuild from chunk_id index mapping
        fed_raw = [
            all_raw[int(cid.split("-")[-1])]
            for cid in survivor_ids
        ]

    expected_companies = {c["company"] for c in fed_raw}
    expected_years = {c["year"] for c in fed_raw}
    expected_numbers = set()
    for c in fed_raw:
        expected_numbers |= extract_numbers(c["text"])

    # Detected in LLM output
    mentioned_companies = extract_companies(answer, expected_companies)
    mentioned_years = extract_years(answer)
    mentioned_numbers = extract_numbers(answer)

    return TestResult(
        test_id=test["test_id"],
        dimension=test["dimension"],
        question=test["question"],
        answer=answer,
        timings=timings,
        expected_companies=expected_companies,
        mentioned_companies=mentioned_companies,
        expected_years=expected_years,
        mentioned_years=mentioned_years,
        expected_numbers=expected_numbers,
        mentioned_numbers=mentioned_numbers,
    )


# ── Aggregation / Reporting ────────────────────────────────────────────

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def aggregate(results: list[TestResult]) -> dict:
    by_dim = defaultdict(list)
    for r in results:
        by_dim[r.dimension].append(r)

    per_dim = {}
    for dim, rs in by_dim.items():
        per_dim[dim] = {
            "n": len(rs),
            "entity_recall": round(_mean([r.entity_recall for r in rs]), 3),
            "entity_purity": round(_mean([r.entity_purity for r in rs]), 3),
            "year_accuracy": round(_mean([r.year_accuracy for r in rs]), 3),
            "numerical_fidelity": round(_mean([r.numerical_fidelity for r in rs]), 3),
        }

    return {
        "n_tests": len(results),
        "entity_recall_mean":      round(_mean([r.entity_recall for r in results]), 3),
        "entity_purity_mean":      round(_mean([r.entity_purity for r in results]), 3),
        "year_accuracy_mean":      round(_mean([r.year_accuracy for r in results]), 3),
        "numerical_fidelity_mean": round(_mean([r.numerical_fidelity for r in results]), 3),
        "avg_gen_time_s":          round(_mean([r.timings.get("final_generation", 0) for r in results]), 2),
        "per_dimension":           per_dim,
    }


def print_report(agg: dict):
    print(f"\n{'='*72}")
    print(f"  Generation Faithfulness — n={agg['n_tests']}")
    print('='*72)
    print(f"  Entity Recall:      {agg['entity_recall_mean']:.3f}   (sources -> output)")
    print(f"  Entity Purity:      {agg['entity_purity_mean']:.3f}   (output entities all in sources)")
    print(f"  Year Accuracy:      {agg['year_accuracy_mean']:.3f}   (output years all in sources)")
    print(f"  Numerical Fidelity: {agg['numerical_fidelity_mean']:.3f}   (output numbers within 5% of source)")
    print(f"  Avg LLM gen time:   {agg['avg_gen_time_s']:.2f}s")
    print('='*72)
    print("\nPer-dimension breakdown:")
    print(f"  {'dimension':24s}  n   recall  purity   year   numfid")
    for dim, d in sorted(agg["per_dimension"].items()):
        print(f"  {dim:24s}  {d['n']:>2}  {d['entity_recall']:.2f}    "
              f"{d['entity_purity']:.2f}    {d['year_accuracy']:.2f}   {d['numerical_fidelity']:.2f}")


def save_report(agg: dict, results: list[TestResult], mode: str, path: Path = REPORT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "mode": mode,
        "summary": agg,
        "per_test": [
            {
                "test_id":              r.test_id,
                "dimension":            r.dimension,
                "question":             r.question,
                "answer":               r.answer,
                "metrics": {
                    "entity_recall":       round(r.entity_recall, 3),
                    "entity_purity":       round(r.entity_purity, 3),
                    "year_accuracy":       round(r.year_accuracy, 3),
                    "numerical_fidelity":  round(r.numerical_fidelity, 3),
                },
                "mentioned_companies":  sorted(r.mentioned_companies),
                "expected_companies":   sorted(r.expected_companies),
                "mentioned_years":      sorted(r.mentioned_years),
                "expected_years":       sorted(r.expected_years),
                "n_numbers_in_output":  len(r.mentioned_numbers),
                "n_numbers_in_sources": len(r.expected_numbers),
                "timings":              r.timings,
            }
            for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to {path}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Generation Faithfulness Eval")
    parser.add_argument("--mode", choices=["oracle", "pipeline"], default="oracle",
                        help="oracle: feed expected_kept | pipeline: run NLI+GAT first")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of tests (debug)")
    parser.add_argument("--backend", default="transformers",
                        help="LLM backend (transformers|ollama|api)")
    args = parser.parse_args()

    # Windows cp1254 → UTF-8 for Turkish output
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    # Load test set
    with open(TESTSET_PATH, encoding="utf-8") as f:
        tests = json.load(f)
    if args.limit:
        tests = tests[: args.limit]
    print(f"Loaded {len(tests)} test cases  (mode={args.mode})")

    # Init LLM + RAG
    from llm_engine import get_engine
    print("Loading LLM...")
    llm = get_engine(args.backend)
    rag = ReliabilityRAG(top_k=20, llm=llm)
    print()

    # Run
    results = []
    for i, test in enumerate(tests):
        prefix = f"[{i+1:>2}/{len(tests)}] {test['dimension']:22s}"
        try:
            r = evaluate_test(rag, test, mode=args.mode)
            results.append(r)
            print(f"{prefix}  rec={r.entity_recall:.2f} pur={r.entity_purity:.2f} "
                  f"yr={r.year_accuracy:.2f} num={r.numerical_fidelity:.2f}   "
                  f"({r.timings.get('final_generation', 0):.1f}s)")
        except Exception as e:
            print(f"{prefix}  FAILED: {type(e).__name__}: {e}")
            continue

    # Report
    agg = aggregate(results)
    print_report(agg)
    save_report(agg, results, mode=args.mode)


if __name__ == "__main__":
    main()
