"""
GAT (Graph Attention Network) + MWIS Dynamic Filtering

This module extends the basic NLI contradiction graph with:
1. Temporal Provenance Embeddings: Year-based features for each node
2. Source Reliability Features: Section type and audit status
3. GAT Layer: Learns attention weights between connected nodes
4. Dynamic MWIS: Uses GAT-refined scores instead of static weights

Architecture:
  Node features = [reliability_weight, year_normalized, section_one_hot, nli_scores]
  Edge features = [contradiction_probability]
  GAT → Updated node embeddings → Consistency score per node
  MWIS with learned scores → Clean subset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from typing import Optional


# ── Node Feature Engineering ────────────────────────────────────────────

SECTION_TYPES = ["finansal", "cevre", "sosyal", "yonetim", "strateji", "genel"]


def build_node_features(
    reliability_weights: list[float],
    years: list[int],
    section_types: list[str],
    nli_contradiction_counts: list[int],
    total_pairs: int,
) -> torch.Tensor:
    """
    Build feature vector for each node.

    Features (per node):
    - reliability_weight (1)
    - year_normalized (1): 0-1 scale, newer = higher
    - section_one_hot (6): one-hot encoding of section type
    - contradiction_ratio (1): fraction of pairs that are contradictions
    = Total: 9 features
    """
    n = len(reliability_weights)
    features = []

    min_year = min(years)
    max_year = max(years)
    year_range = max(max_year - min_year, 1)

    for i in range(n):
        feat = []
        # Reliability weight
        feat.append(reliability_weights[i])
        # Year normalized (newer = 1.0, oldest = 0.0)
        feat.append((years[i] - min_year) / year_range)
        # Section type one-hot
        section_oh = [0.0] * len(SECTION_TYPES)
        if section_types[i] in SECTION_TYPES:
            section_oh[SECTION_TYPES.index(section_types[i])] = 1.0
        feat.extend(section_oh)
        # Contradiction ratio
        feat.append(nli_contradiction_counts[i] / max(total_pairs, 1))

        features.append(feat)

    return torch.tensor(features, dtype=torch.float32)


# ── GAT Layer ───────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """Single-head Graph Attention layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix [N, N] (1 = edge, 0 = no edge)
        Returns:
            Updated node features [N, out_features]
        """
        N = x.size(0)
        h = self.W(x)  # [N, out_features]

        # Compute attention coefficients
        h_i = h.unsqueeze(1).expand(N, N, -1)  # [N, N, out_features]
        h_j = h.unsqueeze(0).expand(N, N, -1)  # [N, N, out_features]
        e = self.leaky_relu(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))  # [N, N]

        # Mask non-edges and self-loops
        mask = (adj > 0).float()
        mask.fill_diagonal_(1.0)  # Allow self-attention
        e = e.masked_fill(mask == 0, float('-inf'))

        # Softmax attention
        alpha = F.softmax(e, dim=-1)  # [N, N]
        alpha = torch.nan_to_num(alpha, 0.0)

        # Aggregate
        out = torch.matmul(alpha, h)  # [N, out_features]
        return F.elu(out)


class GATConsistencyScorer(nn.Module):
    """
    GAT-based consistency scorer.
    Takes node features + adjacency → outputs consistency score per node.
    """

    def __init__(self, in_features: int = 9, hidden: int = 16):
        super().__init__()
        self.gat1 = GATLayer(in_features, hidden)
        self.gat2 = GATLayer(hidden, hidden)
        self.scorer = nn.Sequential(
            nn.Linear(hidden, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Returns consistency score [0, 1] for each node."""
        h = self.gat1(x, adj)
        h = self.gat2(h, adj)
        scores = self.scorer(h).squeeze(-1)  # [N]
        return scores


# ── GAT-Enhanced MWIS ──────────────────────────────────────────────────

class GATFilter:
    """
    Full GAT + MWIS pipeline.

    Three modes:
    1. Heuristic-init (default): no training needed, mimics solve_mwis().
    2. Trained: pass weights_path to load supervised-trained parameters.
       Built by train_gat.py from synthetic_testset.json.
    3. Custom blend: override the GAT-vs-static blend factor (0–1).
       After training, raise from 0.3 → 0.7+ so GAT dominates.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        weights_path: Optional[str] = None,
        blend: float = 0.3,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GATConsistencyScorer().to(self.device)
        self.blend = blend

        if weights_path is not None:
            try:
                state = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                print(f"[GATFilter] Loaded trained weights from {weights_path}")
            except Exception as e:
                print(f"[GATFilter] Failed to load {weights_path}: {e}; falling back to heuristic-init")
                self._init_heuristic_weights()
        else:
            self._init_heuristic_weights()

        self.model.eval()

    @staticmethod
    def build_tensors(
        G: nx.Graph,
        reliability_weights: list[float],
        years: list[int],
        section_types: list[str],
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build (features, adjacency, static_scores) for a single graph instance.
        Shared by inference (compute_scores) and training (train_gat.py).
        """
        n = G.number_of_nodes()
        contradiction_counts = [G.degree(i) for i in range(n)]
        total_pairs = n * (n - 1) // 2

        features = build_node_features(
            reliability_weights, years, section_types,
            contradiction_counts, total_pairs,
        ).to(device)

        adj = torch.zeros(n, n, device=device)
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1.0)
            adj[u, v] = w
            adj[v, u] = w

        # 2026-06-03 (structural fix): match NLIContradictionGraph.solve_mwis
        # exponential decay model. effective_weight = weight × exp(-decay × age).
        # Critical for gat_discriminating Pattern A: old high-reliability vs
        # new lower-reliability cases. Without this, GAT static signal still
        # used old multiplicative formula and disagreed with MWIS — MUST stay
        # aligned so GAT can match or improve over MWIS, not regress.
        import math
        max_year = max(years) if years else 0
        DECAY = 0.15  # data half-life ≈ 4.5 years
        static = []
        for i in range(n):
            age = max(0, max_year - years[i])
            effective_weight = reliability_weights[i] * math.exp(-DECAY * age)
            penalty = 0.1 * contradiction_counts[i]
            raw = effective_weight - penalty
            static.append(max(0.0, min(1.0, raw)))
        static_t = torch.tensor(static, dtype=torch.float32, device=device)

        return features, adj, static_t

    def _init_heuristic_weights(self):
        """
        Initialize GAT weights so the **untrained** model approximates the
        same heuristic NLIContradictionGraph.solve_mwis() uses:
            score ≈ reliability * recency - 0.1 * contradiction_ratio

        BUG NOTE (2026-05-06 ablation): the previous version only set
        scorer[0] but left scorer[2] (8→1) at random init, garbling the
        signal. We now zero ALL scorer params and set both layers as
        identity passes for the first three "channels" (reliability,
        year, contradiction).
        """
        with torch.no_grad():
            # First GAT layer: copy first 9 input features to first 9 of 16
            # output features. Remaining 7 channels stay at default init —
            # they get blended in by gat2 + scorer but contribute zero on
            # the heuristic path because scorer reads only channels 0/1/2.
            self.model.gat1.W.weight.zero_()
            for i in range(min(9, self.model.gat1.W.weight.shape[0])):
                self.model.gat1.W.weight[i, i] = 1.0

            # Second GAT layer: identity on first 16 channels.
            self.model.gat2.W.weight.zero_()
            for i in range(self.model.gat2.W.weight.shape[0]):
                self.model.gat2.W.weight[i, i] = 1.0

            # scorer[0] (Linear 16→8): keep ALL three signals POSITIVE pre-ReLU.
            # The ReLU at scorer[1] kills any negative pre-activation, so we
            # store contradiction as a positive magnitude here and flip the
            # sign at scorer[2] (post-ReLU). Bias adds +0.5 to all three so
            # they survive ReLU even when input feature is zero.
            self.model.scorer[0].weight.data.zero_()
            self.model.scorer[0].bias.data.zero_()
            self.model.scorer[0].weight.data[0, 0] = 2.0   # reliability  → ch0 (+)
            self.model.scorer[0].weight.data[1, 1] = 1.0   # year         → ch1 (+)
            self.model.scorer[0].weight.data[2, 8] = 2.0   # contradiction → ch2 (+, magnitude)
            self.model.scorer[0].bias.data[0] = 0.5
            self.model.scorer[0].bias.data[1] = 0.5
            self.model.scorer[0].bias.data[2] = 0.0  # contradiction has no floor offset

            # scorer[2] (Linear 8→1): combine. Now we negate ch2 here so
            # contradictions DROP the score post-ReLU.
            self.model.scorer[2].weight.data.zero_()
            self.model.scorer[2].bias.data.zero_()
            self.model.scorer[2].weight.data[0, 0] = 1.0   # +reliability signal
            self.model.scorer[2].weight.data[0, 1] = 0.5   # +year signal
            self.model.scorer[2].weight.data[0, 2] = -2.0  # –contradiction penalty (post-ReLU)

    def compute_scores(
        self,
        G: nx.Graph,
        reliability_weights: list[float],
        years: list[int],
        section_types: list[str],
    ) -> list[float]:
        """
        Compute GAT-based consistency scores for all nodes (no-grad inference).

        SKIP CONNECTION (2026-05-06 ablation fix):
        Pure GAT attention-aggregation has a known issue — two nodes joined
        by a contradiction edge get their features averaged, making their
        scores collapse toward each other. We blend the GAT output with the
        raw per-node static signal so that:
            final = self.blend * GAT(x, adj) + (1 - self.blend) * static(x)
        Heuristic-init defaults to 0.3 (static dominates). After supervised
        training, GATFilter(weights_path=..., blend=0.7+).
        """
        features, adj, static_t = self.build_tensors(
            G, reliability_weights, years, section_types, device=self.device,
        )
        with torch.no_grad():
            gat_scores = self.model(features, adj)  # [N], sigmoid output ∈ [0,1]
        blended = self.blend * gat_scores + (1.0 - self.blend) * static_t
        return blended.cpu().tolist()

    def filter_chunks(
        self,
        G: nx.Graph,
        reliability_weights: list[float],
        years: list[int],
        section_types: list[str],
    ) -> list[int]:
        """
        GAT-enhanced MWIS: use learned scores instead of static weights.
        Returns indices of clean (kept) chunks.
        """
        if G.number_of_edges() == 0:
            return list(range(G.number_of_nodes()))

        scores = self.compute_scores(G, reliability_weights, years, section_types)

        # 2026-06-03: Match NLIContradictionGraph.solve_mwis strategy —
        # exact enumeration for n ≤ 12, greedy for larger. Without this,
        # GAT would disagree with MWIS on dense_graph cases (since static
        # signal alignment alone isn't enough; the SELECTION strategy must
        # also be exact to find global optima like [1,3,4,6]).
        nodes = list(range(G.number_of_nodes()))
        if len(nodes) <= 12:
            from nli_graph import NLIContradictionGraph
            adj = {node: set(G.neighbors(node)) for node in nodes}
            score_dict = {node: scores[node] for node in nodes}
            return NLIContradictionGraph._exact_mwis(nodes, score_dict, adj)

        # Greedy MWIS with GAT scores (fallback for n > 12)
        independent_set = []
        remaining = set(nodes)

        while remaining:
            best = max(remaining, key=lambda n: scores[n])
            independent_set.append(best)
            neighbors = set(G.neighbors(best)) & remaining
            remaining.discard(best)
            remaining -= neighbors

        return sorted(independent_set)


def demo():
    """Demo GAT filter with synthetic data."""
    import itertools

    print("=== GAT Filter Demo ===\n")

    # Create a small contradiction graph
    G = nx.Graph()
    n = 6
    reliability = [0.9, 0.6, 0.4, 0.9, 0.6, 0.3]
    years = [2024, 2024, 2024, 2023, 2023, 2022]
    sections = ["finansal", "cevre", "strateji", "finansal", "cevre", "strateji"]

    for i in range(n):
        G.add_node(i, weight=reliability[i], year=years[i])

    # Add some contradiction edges
    contradictions = [(0, 2, 0.95), (1, 2, 0.88), (2, 4, 0.92), (3, 5, 0.85)]
    for u, v, w in contradictions:
        G.add_edge(u, v, weight=w)

    print(f"Graph: {n} nodes, {G.number_of_edges()} edges")
    print(f"Reliability: {reliability}")
    print(f"Years: {years}")
    print(f"Sections: {sections}")
    print(f"Contradictions: {contradictions}")

    # Run GAT filter
    gat = GATFilter()
    clean = gat.filter_chunks(G, reliability, years, sections)

    print(f"\nGAT scores: {[f'{s:.3f}' for s in gat.compute_scores(G, reliability, years, sections)]}")
    print(f"Clean set: {clean}")
    print(f"Removed: {[i for i in range(n) if i not in clean]}")

    # Compare with basic MWIS
    from nli_graph import NLIContradictionGraph
    nli = NLIContradictionGraph.__new__(NLIContradictionGraph)
    basic_clean = []
    remaining = set(range(n))
    max_year = max(years)
    while remaining:
        node_scores = {}
        for node in remaining:
            w = reliability[node]
            y = years[node]
            d = G.degree(node)
            recency = 1.0 + 0.05 * (y - max_year + 10)
            node_scores[node] = w * recency - 0.1 * d
        best = max(remaining, key=lambda n: node_scores[n])
        basic_clean.append(best)
        neighbors = set(G.neighbors(best)) & remaining
        remaining.discard(best)
        remaining -= neighbors

    print(f"\nBasic MWIS clean set: {sorted(basic_clean)}")
    print(f"GAT MWIS clean set:   {clean}")


if __name__ == "__main__":
    demo()
