"""
NLI Contradiction Graph + MIS/MWIS Filtering

Stage 3: Uses a multilingual NLI model to detect contradictions between
         isolated answers (pairwise comparison).
Stage 4: Builds a contradiction graph and applies Maximum (Weighted)
         Independent Set to select the most reliable, consistent subset.

The NLI model classifies pairs as: entailment, neutral, contradiction.
Only "contradiction" pairs create edges in the graph.
"""
import itertools
import time
from typing import Optional

import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Multilingual NLI model — works well for Turkish, fits in 6GB VRAM
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
# 2026-05-06 ablation: at 0.5, NLI missed 5/9 synthetic contradictions
# (Kapsam mismatches, scope abstractions). Lowered to 0.35 with stronger
# numerical-conflict hybrid in build_contradiction_graph(). Test sweep on
# synthetic set: filtering recall jumped from 33% to ~80%.
CONTRADICTION_THRESHOLD = 0.35
NLI_BATCH_SIZE = 32


class NLIContradictionGraph:
    def __init__(self, model_name: str = NLI_MODEL, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading NLI model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Get label mapping
        self.label2id = self.model.config.label2id
        self.contradiction_id = self.label2id.get("contradiction", 2)
        print(f"NLI model loaded. Labels: {self.label2id}")

    def predict_nli_batch(self, premise_list: list[str], hypothesis_list: list[str]) -> list[dict]:
        """
        Predict NLI for a batch of (premise, hypothesis) pairs.
        Returns list of {entailment, neutral, contradiction} probabilities.
        """
        results = []

        for i in range(0, len(premise_list), NLI_BATCH_SIZE):
            batch_premises = premise_list[i:i + NLI_BATCH_SIZE]
            batch_hypotheses = hypothesis_list[i:i + NLI_BATCH_SIZE]

            inputs = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            for j in range(len(batch_premises)):
                prob_dict = {}
                for label, idx in self.label2id.items():
                    prob_dict[label] = probs[j][idx].item()
                results.append(prob_dict)

        return results

    @staticmethod
    def extract_numbers(text: str) -> list[float]:
        """
        Extract numeric values from text for quantitative conflict detection.

        2026-05-06: Added Turkish-thousand-separator pattern (12.500, 1.234.567)
        and bare 4+ digit integers — earlier version only caught %X and X+unit,
        so "12.500 çalışan" / "8.200 çalışan" produced empty number lists and
        the synthetic employee-count test (Test 2) silently no-op'd.
        Years 1900-2100 are excluded to avoid spurious "2022 vs 2023" conflicts.

        2026-05-13: Added Turkish absolute-quantifier words. These let
        "sıfır atık" (claim=0) compare against "45 ton" (45) and
        "%100 yenilenebilir" (claim=100) compare against "%18" (18) —
        greenwashing patterns that NLI alone misses. The mappings are
        intentionally conservative; only unambiguous claims are mapped.
        """
        import re
        patterns = [
            (r'%\s*(\d+[\.,]?\d*)', "percent"),                            # %50, % 30
            (r'(\d+[\.,]?\d*)\s*%', "percent"),                            # 50%, 30%
            (r'(\d+[\.,]?\d*)\s*(milyon|milyar|bin|ton)', "unit"),         # 245 milyon
            (r'\b(\d{1,3}(?:\.\d{3})+)\b', "thousands_sep"),               # 12.500, 1.234.567
            (r'\b(\d{4,})\b', "bare_int"),                                 # 8200, 12500
        ]
        numbers = []
        for p, kind in patterns:
            for m in re.finditer(p, text, re.IGNORECASE):
                raw = m.group(1)
                try:
                    if kind == "thousands_sep":
                        # 12.500 → 12500, 1.234.567 → 1234567
                        val = float(raw.replace(".", ""))
                    else:
                        val = float(raw.replace(',', '.'))
                except ValueError:
                    continue
                # Exclude likely year tokens (1900–2100) — they're rarely
                # the metric being compared and produce false positives.
                if kind == "bare_int" and 1900 <= val <= 2100:
                    continue
                numbers.append(val)

        # Turkish absolute-quantifier words → numeric claims.
        # Word boundary patterns ensure "sıfırlama"-type tokens aren't matched.
        text_lower = text.lower()
        QUANTIFIER_PATTERNS = [
            # Zero-claim: "sıfır", "hiç", "hiçbir"
            (r'\bs[ıi]f[ıi]r\s+(atık|emisyon|kaza|hata|risk|salınım)', 0.0),
            (r'\bhi[çc]bir\s+(atık|emisyon|kaza|hata|risk|salınım)', 0.0),
            (r'\bhi[çc]\s+(atık|emisyon|kaza|salınım)', 0.0),
            # Full-claim: "tamamen", "tamamı", "%100"
            (r'\btamamen\s+(yenilenebilir|geri\s+dön[üu][şs][üu]m|temiz)', 100.0),
            (r'\btamam[ıi]n[ıi]\s+(yenilenebilir|geri\s+dön[üu][şs][üu]m)', 100.0),
            (r'\byüzde\s+yüz\s+(yenilenebilir|temiz)', 100.0),
            (r'\btüm\s+(enerji|atık|emisyon)\s+(yenilenebilir|geri)', 100.0),
        ]
        for pat, val in QUANTIFIER_PATTERNS:
            if re.search(pat, text_lower, re.IGNORECASE):
                numbers.append(val)

        return numbers

    @staticmethod
    def numerical_conflict_score(text_a: str, text_b: str) -> float:
        """
        Detect numerical conflicts: same metric, very different numbers.
        Returns 0-1 score (higher = more likely conflict).
        """
        nums_a = NLIContradictionGraph.extract_numbers(text_a)
        nums_b = NLIContradictionGraph.extract_numbers(text_b)

        if not nums_a or not nums_b:
            return 0.0

        # 2026-05-06 round 4: MIN-ratio not MAX-ratio.
        # MAX-ratio falsely flagged "245M net zarar + 180M faaliyet + 420M
        # giderler" vs "245M zarar" as a 1.71x conflict (420 vs 245). Same
        # for "%50 hedefi → revize %35" vs "güncel %35" (1.43x). The right
        # question is "is there ANY closely-matching number pair across
        # these two chunks?" — if yes, no numerical conflict. We pick the
        # CLOSEST pair (min ratio) and only flag if even that is far apart.
        # Ramp:
        #   min_ratio=1.5 → 0.50,  min_ratio=2.0 → 1.00
        min_ratio = float("inf")
        for a in nums_a:
            for b in nums_b:
                if a == 0 and b == 0:
                    continue
                ratio = max(a, b) / max(min(a, b), 0.01)
                min_ratio = min(min_ratio, ratio)

        if min_ratio == float("inf") or min_ratio <= 1.5:
            return 0.0
        return min((min_ratio - 1.0) / 1.0, 1.0)

    def build_contradiction_graph(
        self,
        texts: list[str],
        reliability_weights: list[float],
        years: list[int],
        threshold: float = CONTRADICTION_THRESHOLD,
    ) -> tuple[nx.Graph, list[tuple[int, int, float]]]:
        """
        Build a contradiction graph from pairwise NLI + numerical conflict detection.

        Uses:
        1. NLI model for semantic contradiction detection
        2. Numerical conflict detector for quantitative mismatches
        3. Temporal penalty for same-topic different-year texts

        Returns (graph, edge_list)
        """
        n = len(texts)

        # Truncate texts for NLI (keep first 256 chars for speed)
        truncated = [t[:256] for t in texts]

        # Generate all pairs
        pairs = list(itertools.combinations(range(n), 2))
        premises = [truncated[i] for i, j in pairs]
        hypotheses = [truncated[j] for i, j in pairs]

        print(f"Running NLI on {len(pairs)} pairs...")
        t0 = time.time()
        nli_results = self.predict_nli_batch(premises, hypotheses)
        elapsed = time.time() - t0
        print(f"NLI done in {elapsed:.1f}s ({len(pairs)/max(elapsed, 0.001):.0f} pairs/sec)")

        # Build graph
        G = nx.Graph()
        for i in range(n):
            G.add_node(i, weight=reliability_weights[i], year=years[i])

        edges = []
        for (i, j), result in zip(pairs, nli_results):
            nli_contradiction = result.get("contradiction", 0)

            # Numerical conflict bonus
            num_conflict = self.numerical_conflict_score(texts[i], texts[j])

            # Hybrid: take the STRONGER signal between semantic NLI and
            # numerical mismatch (instead of weak additive 0.3 weighting).
            # Synthetic-test calibration 2026-05-06: many "scope" /
            # "interdepartmental" cases have weak NLI signal (claim is
            # vague) but obvious numerical disagreement (150K vs 580K ton,
            # %92 satisfaction vs %38 turnover). max() lets either path
            # trigger an edge.
            combined = max(nli_contradiction, num_conflict)
            combined = min(combined, 1.0)

            if combined >= threshold:
                G.add_edge(i, j, weight=combined)
                edges.append((i, j, combined))

        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} contradiction edges")
        return G, edges

    def solve_mwis(self, G: nx.Graph) -> list[int]:
        """
        Solve Maximum Weighted Independent Set (MWIS) on the contradiction graph.

        Uses a greedy heuristic:
        1. Prefer nodes with higher reliability weight
        2. Break ties by preferring more recent year
        3. Among tied, prefer nodes with fewer contradiction edges

        For small graphs (< 30 nodes), this greedy approach works well.
        For large graphs, we could use LP relaxation or sampling.
        """
        if G.number_of_edges() == 0:
            # No contradictions — return all nodes
            return list(G.nodes())

        # Score each node: reliability * recency_bonus - contradiction_penalty
        node_scores = {}
        max_year = max(nx.get_node_attributes(G, "year").values())

        for node in G.nodes():
            weight = G.nodes[node]["weight"]
            year = G.nodes[node]["year"]
            degree = G.degree(node)

            # Recency bonus: newer reports slightly preferred
            recency = 1.0 + 0.05 * (year - max_year + 10)  # +10 to avoid negatives
            # Contradiction penalty: more edges = less trustworthy
            penalty = 0.1 * degree

            node_scores[node] = weight * recency - penalty

        # Greedy MWIS
        independent_set = []
        remaining = set(G.nodes())

        while remaining:
            # Pick the best scoring remaining node
            best = max(remaining, key=lambda n: node_scores.get(n, 0))
            independent_set.append(best)

            # Remove best and all its neighbors
            neighbors = set(G.neighbors(best)) & remaining
            remaining.discard(best)
            remaining -= neighbors

        return sorted(independent_set)

    def filter_chunks(
        self,
        texts: list[str],
        reliability_weights: list[float],
        years: list[int],
        section_types: list[str] = None,
        use_gat: bool = True,
        threshold: float = CONTRADICTION_THRESHOLD,
        gat_weights_path: str = None,
        gat_blend: float = None,
    ) -> tuple[list[int], list[tuple[int, int, float]]]:
        """
        Full pipeline: NLI → Graph → GAT/MWIS → clean indices.

        Args:
            section_types: List of section types per chunk (needed for GAT)
            use_gat: If True, use GAT-enhanced scoring; else basic MWIS
            gat_weights_path: Optional supervised weights for GATFilter.
                              None → heuristic-init.
            gat_blend: Optional inference-time blend override (0..1). None
                       → use GATFilter default (0.3).

        Returns:
            clean_indices: Indices of chunks that passed filtering
            edges: List of (i, j, contradiction_prob) for detected contradictions
        """
        G, edges = self.build_contradiction_graph(
            texts, reliability_weights, years, threshold
        )

        if not edges:
            return list(range(len(texts))), []

        if use_gat and section_types:
            try:
                from gat_filter import GATFilter
                kwargs = {}
                if gat_weights_path is not None:
                    kwargs["weights_path"] = gat_weights_path
                if gat_blend is not None:
                    kwargs["blend"] = gat_blend
                gat = GATFilter(**kwargs)
                clean_indices = gat.filter_chunks(G, reliability_weights, years, section_types)
                method = "GAT+MWIS" + (" (trained)" if gat_weights_path else " (heuristic)")
            except Exception as e:
                print(f"GAT failed ({e}), falling back to basic MWIS")
                clean_indices = self.solve_mwis(G)
                method = "Basic MWIS"
        else:
            clean_indices = self.solve_mwis(G)
            method = "Basic MWIS"

        removed = len(texts) - len(clean_indices)
        print(f"{method}: kept {len(clean_indices)}/{len(texts)} chunks "
              f"(removed {removed} contradictory chunks)")

        return clean_indices, edges


def demo():
    """Demo with synthetic contradictory texts."""
    texts = [
        "Şirketimiz 2023 yılında karbon emisyonlarını %30 azalttı.",
        "2023 yılı itibarıyla karbon salınımlarımız bir önceki yıla göre %30 oranında düşürülmüştür.",
        "Karbon emisyonlarımız 2023'te artış göstermiş olup, hedefe ulaşılamamıştır.",
        "Şirket 2023 yılında toplam 500 bin ton CO2 emisyonu gerçekleştirdi.",
        "Yenilenebilir enerji yatırımlarımız sayesinde 2023'te emisyonlar %30 azalmıştır.",
    ]
    weights = [0.9, 0.6, 0.4, 0.9, 0.6]
    years = [2024, 2024, 2024, 2024, 2024]

    nli = NLIContradictionGraph()

    print("\n--- Pairwise NLI Results ---")
    for i, j in itertools.combinations(range(len(texts)), 2):
        result = nli.predict_nli_batch([texts[i]], [texts[j]])[0]
        c = result.get("contradiction", 0)
        e = result.get("entailment", 0)
        label = max(result, key=result.get)
        if c > 0.3:
            print(f"  [{i}] vs [{j}]: {label} (c={c:.2f}, e={e:.2f}) ***")
        else:
            print(f"  [{i}] vs [{j}]: {label} (c={c:.2f}, e={e:.2f})")

    print("\n--- Full Pipeline ---")
    clean, edges = nli.filter_chunks(texts, weights, years)
    print(f"\nClean set indices: {clean}")
    print("Clean texts:")
    for i in clean:
        print(f"  [{i}] {texts[i]}")


if __name__ == "__main__":
    demo()
