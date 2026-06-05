"""
Evaluation Module for ReliabilityRAG

Measures system performance against success criteria:
1. Inconsistency Filtering Success (target: >=90%)
2. Temporal Consistency Accuracy (target: 100%)
3. Source Fidelity / Hallucination Prevention (target: >=95%) [requires LLM]
4. Algorithmic Latency (target: <5s for 20 chunks)

Runs the NLI + GAT/MWIS pipeline on synthetic test cases
and compares results against ground truth.
"""
import json
import time
from pathlib import Path
from collections import defaultdict

from config import PROJECT_ROOT
from nli_graph import NLIContradictionGraph

TESTSET_PATH = PROJECT_ROOT / "data" / "synthetic_testset.json"


def load_testset(path: Path = TESTSET_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_filtering(
    nli: NLIContradictionGraph,
    tests: list[dict],
    use_gat: bool = True,
    gat_weights_path: str = None,
    gat_blend: float = None,
) -> dict:
    """
    Run NLI + (GAT|MWIS) on each test case and compare with ground truth.

    Args:
        use_gat: True → GAT+MWIS (the contributed approach).
                 False → basic MWIS only (the paper baseline).
        gat_weights_path: Optional path to trained GAT weights. If None,
                          GAT runs in heuristic-init mode.
        gat_blend: Override the GAT-vs-static blend at inference. None = default 0.3.
        Used by ablation_mwis_vs_gat.py to quantify GAT contribution.
    """
    results = {
        "use_gat": use_gat,
        "total_tests": 0,
        "correct_tests": 0,
        "total_contradictory": 0,
        "detected_contradictory": 0,
        "total_clean": 0,
        "kept_clean": 0,
        "false_positives": 0,  # Clean chunks wrongly removed
        "false_negatives": 0,  # Contradictory chunks wrongly kept
        "by_dimension": defaultdict(lambda: {"total": 0, "correct": 0}),
        "latencies": [],
        "details": [],
    }

    for test in tests:
        results["total_tests"] += 1
        dim = test["dimension"]
        results["by_dimension"][dim]["total"] += 1

        chunks = test["chunks"]
        texts = [c["text"] for c in chunks]
        weights = [c["reliability_weight"] for c in chunks]
        years = [c["year"] for c in chunks]
        section_types = [c["section_type"] for c in chunks]
        companies = [c["company"] for c in chunks]  # v5: cross-co + supersession

        expected_kept = set(test["expected_kept"])
        expected_removed = set(test["expected_removed"])

        # Run pipeline
        t0 = time.time()
        actual_kept, edges = nli.filter_chunks(
            texts, weights, years,
            section_types=section_types,
            companies=companies,
            use_gat=use_gat,
            gat_weights_path=gat_weights_path,
            gat_blend=gat_blend,
        )
        latency = time.time() - t0
        results["latencies"].append(latency)

        actual_kept_set = set(actual_kept)
        actual_removed_set = set(range(len(chunks))) - actual_kept_set

        # Count correct removals
        for idx in expected_removed:
            results["total_contradictory"] += 1
            if idx in actual_removed_set:
                results["detected_contradictory"] += 1
            else:
                results["false_negatives"] += 1

        # Count correct keeps
        for idx in expected_kept:
            results["total_clean"] += 1
            if idx in actual_kept_set:
                results["kept_clean"] += 1
            else:
                results["false_positives"] += 1

        # Test-level correctness
        is_correct = (actual_kept_set == expected_kept)
        if is_correct:
            results["correct_tests"] += 1
            results["by_dimension"][dim]["correct"] += 1

        results["details"].append({
            "test_id": test["test_id"],
            "dimension": dim,
            "question": test["question"],
            "expected_kept": sorted(expected_kept),
            "actual_kept": sorted(actual_kept_set),
            "expected_removed": sorted(expected_removed),
            "actual_removed": sorted(actual_removed_set),
            "is_correct": is_correct,
            "latency": latency,
            "contradiction_edges": len(edges),
        })

    return results


def compute_metrics(results: dict) -> dict:
    """Compute final metrics from evaluation results."""
    metrics = {}

    # 1. Inconsistency Filtering Success (target: >=90%)
    if results["total_contradictory"] > 0:
        metrics["filtering_success"] = (
            results["detected_contradictory"] / results["total_contradictory"] * 100
        )
    else:
        metrics["filtering_success"] = 100.0

    # 2. Temporal Consistency (from temporal tests only)
    temporal = results["by_dimension"].get("temporal", {"total": 0, "correct": 0})
    if temporal["total"] > 0:
        metrics["temporal_accuracy"] = temporal["correct"] / temporal["total"] * 100
    else:
        metrics["temporal_accuracy"] = 100.0

    # 3. Clean chunk preservation
    if results["total_clean"] > 0:
        metrics["clean_preservation"] = results["kept_clean"] / results["total_clean"] * 100
    else:
        metrics["clean_preservation"] = 100.0

    # 4. Latency
    if results["latencies"]:
        metrics["avg_latency"] = sum(results["latencies"]) / len(results["latencies"])
        metrics["max_latency"] = max(results["latencies"])
    else:
        metrics["avg_latency"] = 0
        metrics["max_latency"] = 0

    # Overall accuracy
    metrics["overall_accuracy"] = (
        results["correct_tests"] / results["total_tests"] * 100
        if results["total_tests"] > 0 else 0
    )

    # Per-dimension accuracy
    metrics["by_dimension"] = {}
    for dim, data in results["by_dimension"].items():
        metrics["by_dimension"][dim] = (
            data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        )

    return metrics


def print_report(results: dict, metrics: dict):
    """Print evaluation report."""
    print("\n" + "=" * 70)
    print("  ReliabilityRAG - Evaluation Report")
    print("=" * 70)

    print(f"\n  Test cases: {results['total_tests']}")
    print(f"  Correct:    {results['correct_tests']}/{results['total_tests']}")

    print(f"\n  --- Success Criteria ---")
    print(f"  1. Filtering Success:     {metrics['filtering_success']:.1f}%  (target: >=90%)"
          f"  {'PASS' if metrics['filtering_success'] >= 90 else 'FAIL'}")
    print(f"  2. Temporal Accuracy:     {metrics['temporal_accuracy']:.1f}%  (target: 100%)"
          f"  {'PASS' if metrics['temporal_accuracy'] >= 100 else 'FAIL'}")
    print(f"  3. Clean Preservation:    {metrics['clean_preservation']:.1f}%  (target: >=95%)"
          f"  {'PASS' if metrics['clean_preservation'] >= 95 else 'FAIL'}")
    print(f"  4. Avg Latency:           {metrics['avg_latency']:.2f}s  (target: <5s)"
          f"  {'PASS' if metrics['avg_latency'] < 5 else 'FAIL'}")
    print(f"     Max Latency:           {metrics['max_latency']:.2f}s")

    print(f"\n  --- Per Dimension ---")
    for dim, acc in metrics["by_dimension"].items():
        print(f"  {dim}: {acc:.0f}%")

    print(f"\n  --- Confusion Matrix ---")
    print(f"  True Positives  (correctly removed): {results['detected_contradictory']}")
    print(f"  True Negatives  (correctly kept):    {results['kept_clean']}")
    print(f"  False Positives (wrongly removed):   {results['false_positives']}")
    print(f"  False Negatives (wrongly kept):      {results['false_negatives']}")

    # Detail per test
    print(f"\n  --- Test Details ---")
    for d in results["details"]:
        status = "PASS" if d["is_correct"] else "FAIL"
        print(f"  [{status}] [{d['dimension']}] {d['question'][:50]}...")
        if not d["is_correct"]:
            print(f"         Expected kept: {d['expected_kept']}, Got: {d['actual_kept']}")
            print(f"         Expected removed: {d['expected_removed']}, Got: {d['actual_removed']}")

    print("\n" + "=" * 70)


def run_evaluation():
    """Run full evaluation pipeline."""
    print("Loading test set...")
    tests = load_testset()
    print(f"Loaded {len(tests)} test cases")

    print("\nLoading NLI model...")
    nli = NLIContradictionGraph()

    print("\nRunning evaluation...")
    results = evaluate_filtering(nli, tests)
    metrics = compute_metrics(results)

    print_report(results, metrics)

    # Save results
    output_path = PROJECT_ROOT / "data" / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "metrics": metrics}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return metrics


if __name__ == "__main__":
    run_evaluation()
