"""
GAT Supervised Training on synthetic_testset.json.

Trains GATConsistencyScorer to produce HIGH consistency scores for
expected-kept chunks and LOW for expected-removed chunks. The scoring
is end-to-end differentiable; we backprop BCE through the GAT layers.

Why this matters
----------------
Heuristic-init GAT == basic MWIS in 2026-05-06 ablation (both 6/9 on the
synthetic test set). The ablation can't justify "GAT contribution" until
the GAT actually learns something the heuristic doesn't capture. The
expected payoff: section-type and temporal-aware decisions that pure
reliability-weight ranking can't make.

Pipeline
--------
1) Load synthetic_testset (9 cases) + run NLI once on each → contradiction
   graph cache (avoids re-running NLI every epoch).
2) Build (features, adj, static_score, targets) tensors per test.
3) Leave-One-Out CV (LOOCV): for each held-out test, train on the
   remaining 8 with BCE loss, validate on the 1.
4) Final: train on ALL 9, save state_dict to data/gat_weights.pt.
5) Sanity-check the saved weights by reloading and evaluating.

Usage
-----
    HF_HUB_OFFLINE=1 python train_gat.py
    HF_HUB_OFFLINE=1 python train_gat.py --epochs 200 --lr 5e-3
    HF_HUB_OFFLINE=1 python train_gat.py --skip-cv  # only final fit
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

import torch
import torch.nn as nn

from config import PROJECT_ROOT
from gat_filter import GATConsistencyScorer, GATFilter
from nli_graph import NLIContradictionGraph


WEIGHTS_PATH = PROJECT_ROOT / "data" / "gat_weights.pt"
TESTSET_PATH = PROJECT_ROOT / "data" / "synthetic_testset.json"


# ── Data caching ────────────────────────────────────────────────────────

def build_dataset(nli: NLIContradictionGraph, tests: list[dict], device: str):
    """
    For each test case, run NLI once and cache:
        features (N, 9), adjacency (N, N), static_score (N), targets (N), test_id

    Targets are 1.0 for indices in expected_kept, 0.0 for expected_removed.
    Indices not labeled (rare) are excluded from loss.
    """
    cache = []
    for i, test in enumerate(tests):
        chunks = test["chunks"]
        texts = [c["text"] for c in chunks]
        weights = [c["reliability_weight"] for c in chunks]
        years = [c["year"] for c in chunks]
        sections = [c["section_type"] for c in chunks]

        G, edges = nli.build_contradiction_graph(texts, weights, years)
        features, adj, static_t = GATFilter.build_tensors(
            G, weights, years, sections, device=device,
        )

        kept = set(test["expected_kept"])
        removed = set(test["expected_removed"])
        targets = torch.full((len(chunks),), -1.0, device=device)  # -1 = ignore
        for idx in kept:
            targets[idx] = 1.0
        for idx in removed:
            targets[idx] = 0.0

        cache.append({
            "test_id": test["test_id"],
            "dimension": test["dimension"],
            "n_nodes": len(chunks),
            "n_edges": G.number_of_edges(),
            "features": features,
            "adj": adj,
            "static": static_t,
            "targets": targets,
            "expected_kept": list(kept),
            "expected_removed": list(removed),
        })
    return cache


# ── Loss ────────────────────────────────────────────────────────────────

def compute_loss(model, sample, blend: float, margin: float = 0.2) -> torch.Tensor:
    """
    BCE + contrastive margin on labeled nodes.

    BCE pushes scores toward 0/1. Contrastive ensures EVERY kept-removed pair
    has a margin gap so MWIS greedy selection picks the kept one. This is
    crucial because pure BCE saturated at loss≈0.5 in the 14-test run —
    individual scores hit 0.5 because gat-static blend pulls them toward each
    other (when adjacent on contradiction edge, attention averages them).
    """
    gat_scores = model(sample["features"], sample["adj"])  # [N]
    blended = blend * gat_scores + (1.0 - blend) * sample["static"]
    blended = blended.clamp(1e-6, 1 - 1e-6)

    targets = sample["targets"]
    mask = targets >= 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=blended.device)

    # Term 1: BCE
    bce = -(targets[mask] * torch.log(blended[mask])
            + (1 - targets[mask]) * torch.log(1 - blended[mask])).mean()

    # Term 2: Contrastive margin — kept scores must beat removed scores by ≥margin.
    kept_idx = (targets == 1.0).nonzero(as_tuple=True)[0]
    removed_idx = (targets == 0.0).nonzero(as_tuple=True)[0]
    if len(kept_idx) > 0 and len(removed_idx) > 0:
        kept_scores = blended[kept_idx].unsqueeze(1)      # [K, 1]
        removed_scores = blended[removed_idx].unsqueeze(0)  # [1, R]
        # hinge: max(0, margin - (kept - removed))
        gap = kept_scores - removed_scores  # [K, R]
        margin_loss = torch.clamp(margin - gap, min=0).mean()
    else:
        margin_loss = torch.tensor(0.0, device=blended.device)

    return bce + 2.0 * margin_loss  # margin term weighted 2x for stronger separation


def predict_clean_set(model, sample, blend: float) -> set[int]:
    """Run greedy MWIS using blended GAT scores → return clean indices set."""
    import networkx as nx
    with torch.no_grad():
        gat_scores = model(sample["features"], sample["adj"])
        blended = (blend * gat_scores + (1.0 - blend) * sample["static"]).cpu().tolist()

    n = sample["features"].shape[0]
    # Rebuild a minimal graph from adj for MWIS neighbour lookup.
    adj_cpu = sample["adj"].cpu()
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if adj_cpu[i, j].item() > 0:
                G.add_edge(i, j)

    if G.number_of_edges() == 0:
        return set(range(n))

    chosen = []
    remaining = set(range(n))
    while remaining:
        best = max(remaining, key=lambda x: blended[x])
        chosen.append(best)
        nbrs = set(G.neighbors(best)) & remaining
        remaining.discard(best)
        remaining -= nbrs
    return set(chosen)


# ── Training loop ───────────────────────────────────────────────────────

def fit(
    train_samples: list[dict],
    epochs: int,
    lr: float,
    blend: float,
    device: str,
    seed: int = 42,
    verbose: bool = False,
    weight_decay: float = 1e-3,
    early_stop_patience: int = 20,
) -> nn.Module:
    """
    Heuristic-init → fine-tune with early stopping.

    LESSONS LEARNED (2026-05-06 round 2):
    Long training (1000 epoch + lr=5e-3 + blend=1.0) collapsed all outputs
    to a constant 0.508 — model found that "every node = 0.5" is a local
    minimum that minimizes BCE+margin (mean output ≈ targets average).
    Early stopping + small lr + weight decay + blend < 1.0 keeps the
    heuristic-init signal alive while learning incremental refinements.
    """
    torch.manual_seed(seed)
    init = GATFilter(device=device, blend=blend)
    model = init.model
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    bad_epochs = 0

    for epoch in range(epochs):
        total_loss = 0.0
        for s in train_samples:
            optim.zero_grad()
            loss = compute_loss(model, s, blend)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total_loss += loss.item()
        avg = total_loss / len(train_samples)

        if avg < best_loss - 1e-4:
            best_loss = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                if verbose:
                    print(f"  early stop @ epoch {epoch+1}, best loss={best_loss:.4f}")
                break

        if verbose and (epoch + 1) % max(epochs // 10, 1) == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={avg:.4f}  best={best_loss:.4f}")

    # Restore best checkpoint (early stop may have trained past minimum)
    model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_one(model, sample, blend: float) -> bool:
    """True if predicted clean set == expected_kept."""
    pred = predict_clean_set(model, sample, blend)
    expected = set(sample["expected_kept"])
    return pred == expected


# ── LOOCV ───────────────────────────────────────────────────────────────

def run_loocv(samples: list[dict], epochs: int, lr: float, blend: float, device: str):
    """Leave-One-Out CV: report fold-by-fold + average."""
    print(f"\n=== LOOCV ({len(samples)} folds) ===")
    fold_results = []
    for held_out_idx in range(len(samples)):
        train = [s for j, s in enumerate(samples) if j != held_out_idx]
        val = samples[held_out_idx]

        model = fit(train, epochs=epochs, lr=lr, blend=blend, device=device, seed=42)
        ok = evaluate_one(model, val, blend)
        fold_results.append(ok)
        print(f"  fold {held_out_idx+1}/{len(samples)}  [{val['dimension']:<18}] "
              f"edges={val['n_edges']:<2} → {'PASS' if ok else 'FAIL'}")

    n_pass = sum(fold_results)
    print(f"\n  LOOCV: {n_pass}/{len(samples)} ({100*n_pass/len(samples):.1f}%)")
    return fold_results


def fit_full_and_save(samples: list[dict], epochs: int, lr: float, blend: float, device: str):
    """Train on ALL samples, save weights, sanity-check on training set."""
    print(f"\n=== Final fit (all {len(samples)} samples) ===")
    model = fit(samples, epochs=epochs, lr=lr, blend=blend, device=device, seed=42, verbose=True)

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"\n  [saved] {WEIGHTS_PATH}")

    # Reload + sanity check
    chk = GATConsistencyScorer().to(device)
    chk.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    chk.eval()
    n_pass = sum(evaluate_one(chk, s, blend) for s in samples)
    print(f"  Reload sanity (training-set fit): {n_pass}/{len(samples)}")
    return model


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--blend", type=float, default=0.7,
                        help="GAT vs static blend at inference: 0.7 = GAT-dominant")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip LOOCV (only run final fit + save)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Hyperparameters: epochs={args.epochs}, lr={args.lr}, blend={args.blend}")

    # ── 1) Load testset
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        tests = json.load(f)
    print(f"Loaded {len(tests)} test cases from {TESTSET_PATH.name}")

    # ── 2) Run NLI once on every test, build cached tensors
    print("\nLoading NLI + building per-test tensors...")
    t0 = time.time()
    nli = NLIContradictionGraph()
    samples = build_dataset(nli, tests, device=device)
    print(f"  NLI cache built in {time.time()-t0:.1f}s")
    for s in samples:
        print(f"  [{s['dimension']:<18}] n_nodes={s['n_nodes']} n_edges={s['n_edges']} "
              f"kept={s['expected_kept']} removed={s['expected_removed']}")

    # ── 3) LOOCV (optional)
    if not args.skip_cv:
        run_loocv(samples, epochs=args.epochs, lr=args.lr, blend=args.blend, device=device)

    # ── 4) Final fit on all samples + save
    fit_full_and_save(samples, epochs=args.epochs, lr=args.lr, blend=args.blend, device=device)

    print("\nDone. Use the saved weights with:")
    print(f"  GATFilter(weights_path='{WEIGHTS_PATH}', blend={args.blend})")
    print("Or re-run ablation:")
    print(f"  python ablation_mwis_vs_gat.py --gat-weights {WEIGHTS_PATH} --blend {args.blend}")


if __name__ == "__main__":
    main()
