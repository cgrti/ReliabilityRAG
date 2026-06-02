"""
Söylem Değişimi Görselleştirmesi — Discourse Timeline Graph

Şirketlerin yıllar içindeki (2015-2024) söylem değişikliklerini kronolojik
bir çizge üzerinde görselleştirir.

Bu modül, bir şirket × topic için NLI çelişki grafiğini interaktif Plotly
timeline olarak çizer:

  - X ekseni: chunk yılı (2015-2024)
  - Y ekseni: query ile semantic benzerlik (yüksek = topic'le ilgili)
  - Marker rengi: bölüm güvenilirliği (kırmızı=düşük strateji, yeşil=yüksek finansal)
  - Marker sembolü: bölüm türü (finansal=●, çevre=▲, strateji=◆ vb.)
  - Çelişki edge'leri: kırmızı kesik çizgi (NLI tabanlı, yıllar arası söylem
    değişikliği veya kapsam çelişkisi)
  - Hover: chunk text preview + metadata

Standalone CLI:
    python discourse_graph.py --company Akbank --topic "karbon emisyon azaltma"
    → data/discourse_<company>_<slug>.html (browser'da aç)

Gradio'ya entegrasyon için: `build_figure(company, topic_query, top_k)` çağrılır.
"""
import argparse
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import plotly.graph_objects as go

from config import PROJECT_ROOT
from embedder import NumpyVectorSearch
from nli_graph import NLIContradictionGraph


SECTION_SYMBOLS = {
    "finansal": "circle",
    "cevre": "triangle-up",
    "sosyal": "square",
    "yonetim": "diamond",
    "strateji": "x",
    "genel": "circle-open",
}

SECTION_COLORS_BY_RELIABILITY = {
    0.9: "#2ca02c",  # finansal — yeşil (güvenilir)
    0.7: "#9467bd",  # yonetim — mor
    0.6: "#1f77b4",  # cevre — mavi
    0.5: "#ff7f0e",  # sosyal/genel — turuncu
    0.4: "#d62728",  # strateji — kırmızı (en az güvenilir)
}


def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s.lower()).strip("_")[:40]


def build_edge_table(results: list[dict], edges: list[tuple[int, int, float]],
                      top_n: int = 15) -> list[list[str]]:
    """
    Build rows for the "top contradictory pairs" table.
    Each row: [year_a, year_b, prob, company, section_a→section_b, preview_a, preview_b]
    Sorted by contradiction probability descending.
    """
    rows = []
    for i, j, prob in sorted(edges, key=lambda e: -e[2])[:top_n]:
        a = results[i]
        b = results[j]
        rows.append([
            f"{a['year']} ↔ {b['year']}",
            f"{prob:.2f}",
            f"{a['section_type']} → {b['section_type']}",
            (a["text"][:90] + "…").replace("\n", " "),
            (b["text"][:90] + "…").replace("\n", " "),
        ])
    return rows


def build_figure(
    company: str,
    topic_query: str,
    top_k: int = 30,
    searcher: NumpyVectorSearch = None,
    nli: NLIContradictionGraph = None,
) -> tuple[go.Figure, dict]:
    """
    Build the Plotly discourse-timeline figure.

    Returns:
        figure, stats_dict (with retrieval+nli counts, edge count, time range,
        and 'edge_rows' for the contradiction-pair table).
    """
    if searcher is None:
        searcher = NumpyVectorSearch()
    if nli is None:
        nli = NLIContradictionGraph()

    # ── 1) Retrieve top-K for this company × topic ─────────────────────
    results = searcher.search(topic_query, top_k=top_k, company=company)
    if not results:
        # Empty figure with a friendly message
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ {company} için '{topic_query}' sorgusunda chunk bulunamadı.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(title="Söylem Değişimi", template="plotly_white")
        return fig, {"retrieved": 0, "edges": 0}

    texts = [r["text"] for r in results]
    weights = [r["reliability_weight"] for r in results]
    years = [r["year"] for r in results]

    # ── 2) NLI contradiction graph ────────────────────────────────────
    G, edges = nli.build_contradiction_graph(texts, weights, years)
    # edges is list of (i, j, contradiction_prob)

    # ── 3) Compose Plotly figure ──────────────────────────────────────
    fig = go.Figure()

    # 3a) Contradiction edges first (so they sit behind markers)
    for i, j, prob in edges:
        a = results[i]
        b = results[j]
        fig.add_trace(go.Scatter(
            x=[a["year"], b["year"]],
            y=[a["score"], b["score"]],
            mode="lines",
            line=dict(color=f"rgba(214, 39, 40, {min(0.9, 0.3 + prob*0.6)})",
                      width=1.5, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
            name=f"Çelişki ({prob:.2f})",
        ))

    # 3b) Group markers by section type (one trace per section, for clean legend)
    by_section: dict[str, list[dict]] = {}
    for r in results:
        by_section.setdefault(r["section_type"], []).append(r)

    for section, chunks in sorted(by_section.items()):
        fig.add_trace(go.Scatter(
            x=[c["year"] for c in chunks],
            y=[c["score"] for c in chunks],
            mode="markers",
            marker=dict(
                size=14,
                symbol=SECTION_SYMBOLS.get(section, "circle"),
                color=[SECTION_COLORS_BY_RELIABILITY.get(c["reliability_weight"], "#7f7f7f")
                       for c in chunks],
                line=dict(width=1, color="white"),
            ),
            name=f"{section} (n={len(chunks)})",
            hovertemplate=(
                "<b>%{customdata[0]} · %{x}</b><br>"
                "Bölüm: %{customdata[1]} · Güv: %{customdata[2]}<br>"
                "Benzerlik: %{y:.3f}<br>"
                "<i>%{customdata[3]}…</i><extra></extra>"
            ),
            customdata=[[
                c["company"], c["section_type"],
                c["reliability_weight"], c["text"][:160].replace("\n", " "),
            ] for c in chunks],
        ))

    # 3c) Layout
    min_year = min(years) - 1
    max_year = max(years) + 1
    fig.update_layout(
        title=dict(
            text=f"<b>{company}</b> — Söylem Değişimi Timeline'ı<br>"
                 f"<span style='font-size:12px;color:#666'>"
                 f"Sorgu: <i>{topic_query}</i> · {len(results)} chunk · "
                 f"{len(edges)} çelişki edge'i</span>",
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="Yıl",
            range=[min_year, max_year],
            tickmode="linear", dtick=1,
            gridcolor="#eee",
        ),
        yaxis=dict(
            title="Konuyla Benzerlik (cosine similarity)",
            gridcolor="#eee",
        ),
        template="plotly_white",
        height=560,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5,
            title="Bölüm türü (renk = güvenilirlik: 🟢 finansal > 🔵 çevre > 🔴 strateji)",
        ),
        hovermode="closest",
        margin=dict(t=80, l=60, r=20, b=80),
        # Gradio gr.Plot container'ına fit etmesi için. Hardcoded height olmadan
        # bazı Gradio 6.x sürümleri Plotly figure'u silently gizli yükseklikte
        # render ediyor — autosize bu sorunu çözer.
        autosize=True,
    )

    stats = {
        "retrieved": len(results),
        "edges": len(edges),
        "year_range": [min(years), max(years)],
        "sections": {s: len(c) for s, c in by_section.items()},
        "edge_rows": build_edge_table(results, edges, top_n=15),
    }
    return fig, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", required=True, help="e.g. Akbank")
    parser.add_argument("--topic", required=True, help="e.g. 'karbon emisyon azaltma hedefi'")
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    print(f"Company: {args.company}")
    print(f"Topic:   {args.topic}")
    print(f"Top-K:   {args.top_k}")
    print()

    fig, stats = build_figure(args.company, args.topic, args.top_k)
    print(f"Stats: {stats}")

    slug = _slugify(f"{args.company}_{args.topic}")
    out_path = PROJECT_ROOT / "data" / f"discourse_{slug}.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"\n[saved] {out_path}")
    print("Browser'da aç:")
    print(f"  start {out_path}")


if __name__ == "__main__":
    main()
