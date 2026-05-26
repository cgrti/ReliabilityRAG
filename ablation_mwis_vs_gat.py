"""
Ablation: MWIS-only vs MWIS+GAT
Quantitative evidence for the GAT contribution in ReliabilityRAG.

Methodology
-----------
Runs the same evaluation pipeline (`evaluation.evaluate_filtering`) twice on the
9-case synthetic test set (3 temporal + 3 scope + 3 interdepartmental):

    Run A: use_gat=False   →  paper baseline (basic MWIS, greedy heuristic)
    Run B: use_gat=True    →  contributed approach (GAT-scored MWIS)

Same NLI model instance is reused, same test cases, same threshold — only the
filtering layer changes. Latency measured per-test, averaged.

Output
------
    1) Side-by-side metrics table (filtering_success, temporal_accuracy, etc.)
    2) Per-test diff: which mode got which test right
    3) JSON dump → data/ablation_results.json   (for thesis tables)

Run
---
    python ablation_mwis_vs_gat.py
    python ablation_mwis_vs_gat.py --gat-weights data/gat_weights.pt --blend 0.7
"""
import argparse
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
from evaluation import (
    load_testset,
    evaluate_filtering,
    compute_metrics,
)
from nli_graph import NLIContradictionGraph


OUTPUT_PATH = PROJECT_ROOT / "data" / "ablation_results.json"


def _classify_per_test(details_a: list[dict], details_b: list[dict]) -> list[dict]:
    """
    Per-test verdict: which mode was correct?
    Output rows pair test_id-aligned details from MWIS and GAT runs.
    """
    by_id_a = {d["test_id"]: d for d in details_a}
    rows = []
    for db in details_b:
        da = by_id_a.get(db["test_id"], {})
        a_ok = da.get("is_correct", False)
        b_ok = db.get("is_correct", False)

        if a_ok and b_ok:
            verdict = "both_ok"
        elif a_ok and not b_ok:
            verdict = "mwis_only_ok"   # GAT regressed
        elif not a_ok and b_ok:
            verdict = "gat_only_ok"    # GAT contributed
        else:
            verdict = "both_failed"

        rows.append({
            "test_id": db["test_id"],
            "dimension": db["dimension"],
            "question": db["question"][:60],
            "expected_kept": db["expected_kept"],
            "mwis_kept": da.get("actual_kept", []),
            "gat_kept": db.get("actual_kept", []),
            "mwis_correct": a_ok,
            "gat_correct": b_ok,
            "verdict": verdict,
            "mwis_latency": round(da.get("latency", 0.0), 3),
            "gat_latency": round(db.get("latency", 0.0), 3),
        })
    return rows


def _format_diff_table(metrics_a: dict, metrics_b: dict) -> str:
    """Format a side-by-side metrics table in markdown."""
    rows = [
        ("Filtreleme Başarısı (recall)", "filtering_success", "%", ">=90"),
        ("Temporal Doğruluk",            "temporal_accuracy", "%", "100"),
        ("Temiz Chunk Korunması",        "clean_preservation", "%", ">=95"),
        ("Genel Doğruluk",               "overall_accuracy", "%", "—"),
        ("Ortalama Gecikme",             "avg_latency",  "s", "<5"),
        ("Maks Gecikme",                 "max_latency",  "s", "—"),
    ]
    out = []
    out.append(f"| {'Metrik':<32} | {'MWIS-only':>10} | {'MWIS+GAT':>10} | {'Δ':>8} | {'Hedef':>6} |")
    out.append(f"|{'-'*34}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*8}|")
    for label, key, unit, target in rows:
        a = metrics_a.get(key, 0.0)
        b = metrics_b.get(key, 0.0)
        delta = b - a
        sign = "+" if delta >= 0 else ""
        out.append(
            f"| {label:<32} | "
            f"{a:>8.2f}{unit:<2}| {b:>8.2f}{unit:<2}| "
            f"{sign}{delta:>6.2f} | {target:>6} |"
        )
    return "\n".join(out)


def _format_verdict_summary(rows: list[dict]) -> str:
    """Per-verdict counts + per-dimension breakdown."""
    from collections import Counter
    verdicts = Counter(r["verdict"] for r in rows)
    by_dim = {}
    for r in rows:
        by_dim.setdefault(r["dimension"], Counter())[r["verdict"]] += 1

    out = []
    out.append("\nGenel Verdikt Dağılımı:")
    out.append(f"  both_ok        : {verdicts.get('both_ok', 0):>2}  (her ikisi de geçti)")
    out.append(f"  gat_only_ok    : {verdicts.get('gat_only_ok', 0):>2}  (GAT katkısı — yalnız GAT doğru)")
    out.append(f"  mwis_only_ok   : {verdicts.get('mwis_only_ok', 0):>2}  (GAT regresyon — yalnız MWIS doğru)")
    out.append(f"  both_failed    : {verdicts.get('both_failed', 0):>2}  (ikisi de kaçırdı)")
    out.append("\nBoyut Bazında:")
    for dim, counter in by_dim.items():
        ok_g = counter.get("gat_only_ok", 0) + counter.get("both_ok", 0)
        ok_m = counter.get("mwis_only_ok", 0) + counter.get("both_ok", 0)
        total = sum(counter.values())
        out.append(f"  {dim:<22} → MWIS {ok_m}/{total}  GAT {ok_g}/{total}")
    return "\n".join(out)


def _format_failure_details(rows: list[dict]) -> str:
    """List the test cases where the two modes disagree."""
    diffs = [r for r in rows if r["verdict"] in ("gat_only_ok", "mwis_only_ok")]
    if not diffs:
        return "\n(MWIS ile GAT'ın ayrıştığı test yok — ikisi de aynı sonucu verdi.)"
    out = ["\nMod Ayrışan Testler:"]
    for r in diffs:
        marker = "GAT KAZANDI" if r["verdict"] == "gat_only_ok" else "GAT REGRESYON"
        out.append(
            f"  [{marker}] [{r['dimension']}] {r['question']}...\n"
            f"     expected: {r['expected_kept']}\n"
            f"     MWIS    : {r['mwis_kept']}  ({'OK' if r['mwis_correct'] else 'X'})\n"
            f"     GAT     : {r['gat_kept']}  ({'OK' if r['gat_correct'] else 'X'})"
        )
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gat-weights", type=str, default=None,
                        help="Path to trained GAT weights (.pt). Default: heuristic-init.")
    parser.add_argument("--blend", type=float, default=None,
                        help="GAT-vs-static blend at inference (0..1). Default: 0.3 (heuristic).")
    args = parser.parse_args()

    print("=" * 70)
    print("  ABLATION: MWIS-only vs MWIS+GAT")
    if args.gat_weights:
        print(f"  GAT mode: TRAINED  weights={args.gat_weights}  blend={args.blend or 0.3}")
    else:
        print(f"  GAT mode: HEURISTIC-INIT  blend={args.blend or 0.3}")
    print("=" * 70)

    print("\n[1/4] Loading test set...")
    tests = load_testset()
    print(f"      → {len(tests)} test cases loaded")

    print("\n[2/4] Loading NLI model (single instance, reused for both runs)...")
    t0 = time.time()
    nli = NLIContradictionGraph()
    print(f"      → NLI ready in {time.time()-t0:.1f}s")

    print("\n[3/4] Run A: use_gat=False (basic MWIS — paper baseline)")
    print("-" * 70)
    t0 = time.time()
    results_mwis = evaluate_filtering(nli, tests, use_gat=False)
    metrics_mwis = compute_metrics(results_mwis)
    elapsed_a = time.time() - t0
    print(f"      → Run A done in {elapsed_a:.1f}s")

    print("\n[4/4] Run B: use_gat=True (MWIS+GAT — contributed)")
    print("-" * 70)
    t0 = time.time()
    results_gat = evaluate_filtering(
        nli, tests, use_gat=True,
        gat_weights_path=args.gat_weights,
        gat_blend=args.blend,
    )
    metrics_gat = compute_metrics(results_gat)
    elapsed_b = time.time() - t0
    print(f"      → Run B done in {elapsed_b:.1f}s")

    # ── Report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  KARŞILAŞTIRMA TABLOSU")
    print("=" * 70 + "\n")
    print(_format_diff_table(metrics_mwis, metrics_gat))

    diff_rows = _classify_per_test(results_mwis["details"], results_gat["details"])
    print(_format_verdict_summary(diff_rows))
    print(_format_failure_details(diff_rows))

    # ── Persist ─────────────────────────────────────────────────────────
    payload = {
        "summary": {
            "total_tests": len(tests),
            "mwis_correct": results_mwis["correct_tests"],
            "gat_correct": results_gat["correct_tests"],
            "mwis_run_seconds": round(elapsed_a, 2),
            "gat_run_seconds": round(elapsed_b, 2),
        },
        "metrics_mwis": metrics_mwis,
        "metrics_gat": metrics_gat,
        "per_test": diff_rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[saved] {OUTPUT_PATH}")
    print("=" * 70)
    print(f"  ÖZET: MWIS {results_mwis['correct_tests']}/{len(tests)}  vs  "
          f"GAT {results_gat['correct_tests']}/{len(tests)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
