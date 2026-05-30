"""
ReliabilityRAG Gradio Demo
Interactive UI for the 6-stage pipeline:
  Retrieval -> (Isolated Answering) -> NLI -> MWIS+GAT -> Final Generation

Launch:
  python app.py                      # full mode (loads LLM at startup)
  RELIABILITY_RAG_NO_LLM=1 python app.py   # stub mode (no LLM, raw chunks)
"""
import os
import sys
import time

# UTF-8 safe output on Windows consoles (cp1254 default).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import gradio as gr

from rag_pipeline import ReliabilityRAG
from llm_engine import get_engine
from discourse_graph import build_figure as build_discourse_figure


# ── State: loaded once at startup ───────────────────────────────────────

print("=" * 70)
print("  ReliabilityRAG Gradio App — starting")
print("=" * 70)

USE_LLM = os.environ.get("RELIABILITY_RAG_NO_LLM", "0") != "1"

_llm = None
if USE_LLM:
    print("[loading LLM...]")
    t0 = time.time()
    _llm = get_engine("transformers")
    print(f"[LLM loaded in {time.time()-t0:.1f}s]")
else:
    print("[stub mode — no LLM, raw chunks will be returned]")

_rag = ReliabilityRAG(top_k=20, llm=_llm)

print("[warming up searcher + NLI...]")
t0 = time.time()
_ = _rag.searcher     # loads embeddings + chunks
_ = _rag.nli          # loads mDeBERTa
print(f"[warmed in {time.time()-t0:.1f}s]")

# Unique company + year lists for dropdowns (exact strings from metadata).
_companies = sorted({c["company"] for c in _rag.searcher.chunks})
_years = sorted({c["year"] for c in _rag.searcher.chunks})
print(f"[{len(_companies)} companies, years {_years[0]}-{_years[-1]}]")
print("=" * 70)


# ── Core query function ─────────────────────────────────────────────────

ALL = "(Hepsi)"


def _metrics_md(retrieved, kept, removed, edges, total=None):
    s = (
        f"| Metrik | Değer |\n"
        f"|--------|------:|\n"
        f"| Getirilen (retrieved) | **{retrieved}** |\n"
        f"| Temiz kümede kalan    | **{kept}** |\n"
        f"| Çıkarılan (removed)   | **{removed}** |\n"
        f"| Çelişki kenarı        | **{edges}** |\n"
    )
    if total is not None:
        s += f"| Toplam süre           | **{total:.1f}s** |\n"
    return s


def _sources_md(clean_chunks):
    if not clean_chunks:
        return "_(Hiç kaynak kalmadı)_"
    parts = []
    for i, c in enumerate(clean_chunks[:10], 1):
        parts.append(
            f"### {i}. {c.company} — {c.year}\n"
            f"**Bölüm:** `{c.section_type}` · "
            f"**Güvenilirlik:** `{c.reliability_weight}` · "
            f"**Benzerlik:** `{c.score:.3f}`\n\n"
            f"> {c.text[:200]}...\n"
        )
    return "\n---\n\n".join(parts)


def run_query_stream(question, company, year, top_k, use_nli_filter, use_isolated):
    """
    Generator: streams pipeline progress through 5 phases into the UI.

    Yields (answer, metrics_md, timings_rows, sources_md) tuples:
      1. After retrieval    — metrics filled, answer shows status
      2. After NLI+GAT      — sources filled, answer shows status
      3. During LLM stream  — answer grows token by token
      4. Final              — complete timings including total
    """
    if not question or not question.strip():
        yield "Lütfen bir soru girin.", "", [], ""
        return

    company_filter = None if company == ALL else company
    year_filter = None if year == ALL else int(year)

    # ── Stage 1: Retrieval ──────────────────────────────────────────
    yield "⏳ Retrieval başladı...", "", [], ""
    t0 = time.time()
    try:
        chunks = _rag.retrieve(question, int(top_k), company_filter, year_filter)
    except Exception as e:
        yield f"[Retrieval HATA] {e}", "", [], ""
        return
    retrieval_time = time.time() - t0

    # Early preview with retrieved info, no filtering yet
    yield (
        f"⏳ Retrieval tamam ({len(chunks)} chunk, {retrieval_time:.1f}s). "
        f"NLI + MWIS + GAT filtrelemesi başlıyor...",
        _metrics_md(len(chunks), len(chunks), 0, 0),
        [["retrieval", f"{retrieval_time:.2f}"]],
        "",
    )

    # ── Stage 3+4+5: NLI + MWIS + GAT ───────────────────────────────
    clean_chunks = chunks
    contradiction_edges = []
    nli_time = 0.0
    if use_nli_filter and len(chunks) > 1:
        try:
            t0 = time.time()
            clean_indices, contradiction_edges = _rag.nli.filter_chunks(
                [c.text for c in chunks],
                [c.reliability_weight for c in chunks],
                [c.year for c in chunks],
                section_types=[c.section_type for c in chunks],
                use_gat=True,
            )
            clean_chunks = [chunks[i] for i in clean_indices]
            nli_time = time.time() - t0
        except Exception as e:
            yield f"[NLI HATA] {e}", "", [], ""
            return

    sources_md = _sources_md(clean_chunks)
    metrics_partial = _metrics_md(
        len(chunks), len(clean_chunks),
        len(chunks) - len(clean_chunks), len(contradiction_edges),
    )
    timings_partial = [
        ["retrieval", f"{retrieval_time:.2f}"],
        ["nli_graph_filter", f"{nli_time:.2f}"],
    ]
    yield (
        f"⏳ Filtreleme tamam. {len(clean_chunks)}/{len(chunks)} chunk kaldı. "
        f"LLM cevap üretiyor...",
        metrics_partial,
        timings_partial,
        sources_md,
    )

    # ── Stage 6: Final Generation ───────────────────────────────────
    if _llm is None:
        # Stub mode — raw sources
        stub_answer = _rag.final_answer(question, clean_chunks)
        total = retrieval_time + nli_time
        yield (
            stub_answer,
            _metrics_md(
                len(chunks), len(clean_chunks),
                len(chunks) - len(clean_chunks),
                len(contradiction_edges), total=total,
            ),
            timings_partial + [["TOTAL", f"{total:.2f}"]],
            sources_md,
        )
        return

    # Build final prompt
    from llm_engine import FINAL_ANSWER_PROMPT
    from config import get_profile

    source_blocks = []
    for i, c in enumerate(clean_chunks[:5]):
        source_blocks.append(
            f"Kaynak {i+1} [{c.company} {c.year} | {c.section_type} | "
            f"güvenilirlik={c.reliability_weight}]:\n{c.text[:600]}"
        )
    sources_text = "\n\n---\n\n".join(source_blocks)
    prompt = FINAL_ANSWER_PROMPT.format(sources=sources_text, question=question)
    max_tokens = get_profile()["final_answer_max_tokens"]

    t0 = time.time()
    partial = ""
    try:
        for partial in _llm.generate_stream(prompt, max_tokens=max_tokens):
            elapsed = time.time() - t0
            yield (
                partial,
                metrics_partial,
                timings_partial + [["final_generation (streaming)", f"{elapsed:.1f}"]],
                sources_md,
            )
    except Exception as e:
        yield f"[LLM HATA] {e}\n\nKısmi cevap: {partial}", metrics_partial, timings_partial, sources_md
        return

    final_time = time.time() - t0
    total_time = retrieval_time + nli_time + final_time
    yield (
        partial or "(Cevap boş)",
        _metrics_md(
            len(chunks), len(clean_chunks),
            len(chunks) - len(clean_chunks),
            len(contradiction_edges), total=total_time,
        ),
        timings_partial + [
            ["final_generation", f"{final_time:.2f}"],
            ["TOTAL", f"{total_time:.2f}"],
        ],
        sources_md,
    )


# ── Build UI ────────────────────────────────────────────────────────────

DESCRIPTION = """
# ReliabilityRAG — Multi-Document Çelişki Tespiti

**248 Türk kurumsal entegre raporu** (63 şirket, 2015-2024, 182K chunk) üzerinde
çalışan, **GAT tabanlı dinamik filtreleme** ile bilgi çelişkilerini otomatik tespit
eden RAG sistemi.

**6 aşamalı pipeline:** Retrieval → (Isolated Answering) → NLI Contradiction Graph
→ MWIS Filtering → GAT Dynamic Filtering → Final Generation

_CSE496 Bitirme Projesi — Danışman: Prof. Yusuf Sinan Akgül_
"""

EXAMPLES = [
    ["Akbank'ın karbon emisyon azaltma hedefleri nelerdir?", "Akbank", ALL],
    ["Garanti BBVA'nın toplam geliri ne kadar?", "GarantiBBVA", ALL],
    ["Çimento şirketlerinin sürdürülebilirlik stratejileri nelerdir?", "NuhCimento", ALL],
    ["Kadın istihdamı ve fırsat eşitliği politikaları", ALL, ALL],
    ["Yenilenebilir enerji yatırımları", ALL, "2023"],
    ["İş sağlığı ve güvenliği kaza istatistikleri", ALL, ALL],
]


def _plot_to_html(fig, height: int = 600) -> str:
    """
    Convert Plotly figure to embeddable HTML string. Bypasses gr.Plot widget
    which has rendering issues on Gradio 6.11 + Plotly 5.24 (2026-05-31 bug:
    Plot container renders at 0px even with autosize/CSS fixes; standalone
    HTML from fig.to_html() works perfectly). gr.HTML is the bulletproof path.
    """
    return fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        default_height=height,
        default_width="100%",
    )


def run_discourse(company_sel, topic, k):
    """Wraps build_discourse_figure for Gradio — reuses already-loaded NLI+searcher.
    Returns HTML string (not Plot figure) due to Gradio 6.11 gr.Plot bug."""
    if not company_sel or company_sel == ALL:
        return (
            "<div style='padding:40px;text-align:center;color:#888'>"
            "Lütfen bir şirket seçin.</div>",
            "_(Şirket seçilmedi)_",
            [],
        )
    if not topic or not topic.strip():
        topic = "sürdürülebilirlik performansı"
    try:
        fig, stats = build_discourse_figure(
            company_sel, topic, top_k=int(k),
            searcher=_rag.searcher, nli=_rag.nli,
        )
        info = (
            f"**{company_sel}** · sorgu: *{topic}* · "
            f"getirilen **{stats['retrieved']}** chunk · "
            f"çelişki **{stats['edges']}** edge · "
            f"yıl aralığı: {stats.get('year_range', '?')}\n\n"
            f"Bölüm dağılımı: " +
            ", ".join(f"`{s}`={n}" for s, n in stats.get("sections", {}).items()) +
            f"\n\n_En çelişkili 15 çift aşağıda; tam liste {stats['edges']} edge._"
        )
        return _plot_to_html(fig), info, stats.get("edge_rows", [])
    except Exception as e:
        return (
            f"<div style='padding:40px;text-align:center;color:#c00'>"
            f"<b>Hata:</b> {e}</div>",
            f"❌ {e}",
            [],
        )


with gr.Blocks(title="ReliabilityRAG") as app:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        # ────────────────────────────────────────────────────────────────
        with gr.Tab("🔍 Sorgulama"):
            with gr.Row():
                # ── Left: input ────────────────────────────────────────
                with gr.Column(scale=2):
                    question = gr.Textbox(
                        label="Soru",
                        placeholder="Ör: Akbank'ın karbon emisyon azaltma hedefleri nelerdir?",
                        lines=3,
                    )
                    with gr.Row():
                        company = gr.Dropdown(
                            choices=[ALL] + _companies,
                            value=ALL,
                            label="Şirket filtresi",
                        )
                        year = gr.Dropdown(
                            choices=[ALL] + [str(y) for y in _years],
                            value=ALL,
                            label="Yıl filtresi",
                        )

                    with gr.Accordion("Gelişmiş ayarlar", open=False):
                        top_k = gr.Slider(5, 50, value=20, step=5, label="Top-K (getirilen chunk sayısı)")
                        use_nli_filter = gr.Checkbox(
                            value=True,
                            label="NLI + MWIS + GAT çelişki filtresi (ÖNERİLİR)",
                        )
                        use_isolated = gr.Checkbox(
                            value=False,
                            label="Isolated answering (her chunk için ayrı LLM cevabı — ÇOK YAVAŞ)",
                        )

                    submit_btn = gr.Button("🔍 Sorgula", variant="primary", size="lg")

                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[question, company, year],
                        label="Örnek sorular (tıkla, doldur, Sorgula'ya bas)",
                    )

                # ── Right: output ──────────────────────────────────────
                with gr.Column(scale=3):
                    answer_out = gr.Textbox(
                        label="📝 Cevap",
                        lines=14,
                    )
                    metrics_out = gr.Markdown(label="📊 Pipeline metrikleri")
                    timings_out = gr.Dataframe(
                        headers=["Aşama", "Süre (s)"],
                        label="⏱ Aşama süreleri",
                        datatype=["str", "str"],
                        interactive=False,
                    )

            with gr.Accordion("📚 Kaynaklar (filtreleme sonrası temiz küme)", open=False):
                sources_out = gr.Markdown()

        # ────────────────────────────────────────────────────────────────
        with gr.Tab("📈 Söylem Timeline"):
            gr.Markdown(
                "### Şirketin yıllar içindeki söylem değişimi\n"
                "Bir şirket seç + bir konu yaz → o konudaki chunk'ları yıllar boyunca "
                "**benzerlik × yıl** ekseninde göster. Kırmızı kesik çizgi = NLI'nin "
                "tespit ettiği çelişki edge'i (yıllar arası tutarsız iddialar). "
                "_(Saliha Hoca, 2026-05-10)_\n\n"
                "**Hızlı örnekler:** `Akbank` + *karbon emisyon azaltma hedefi* · "
                "`GarantiBBVA` + *kadın çalışan oranı* · "
                "`NuhCimento` + *yenilenebilir enerji yatırımı* · "
                "`AdanaCimento` + *karbon emisyonu*"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    disc_company = gr.Dropdown(
                        choices=_companies,
                        value=_companies[0] if _companies else None,
                        label="Şirket",
                    )
                    disc_topic = gr.Textbox(
                        label="Konu / Sorgu",
                        placeholder="Ör: karbon emisyon azaltma hedefi",
                        lines=2,
                    )
                    disc_k = gr.Slider(10, 60, value=30, step=5, label="Top-K (chunk sayısı)")
                    disc_btn = gr.Button("📈 Çizgeyi Üret", variant="primary", size="lg")
                    # NOTE: gr.Examples removed 2026-05-31 — Gradio 6.11 Dataset.svelte:228
                    # rendering bug with string columns. Examples documented in markdown above
                    # for users to type manually. Doesn't affect core feature functionality.
                with gr.Column(scale=3):
                    # Using gr.HTML instead of gr.Plot due to Gradio 6.11 + Plotly 5.24
                    # rendering bug (Plot container 0px height). HTML embed bypasses
                    # the widget — standalone fig.to_html() verified working.
                    disc_plot = gr.HTML(
                        value="<div style='padding:40px;text-align:center;color:#888;"
                              "min-height:560px'>📈 Çizgeyi üretmek için butona tıkla.</div>",
                        label="Söylem Timeline",
                    )
                    disc_info = gr.Markdown()

            with gr.Accordion("⚡ En Çelişkili Çiftler (NLI tabanlı, top 15)", open=True):
                disc_edges = gr.Dataframe(
                    headers=["Yıl Çifti", "Çelişki Olasılığı", "Bölüm Akışı",
                             "Chunk A (önizleme)", "Chunk B (önizleme)"],
                    datatype=["str", "str", "str", "str", "str"],
                    label="Söylem değişikliği kanıtı: hangi chunk'lar hangi yıllar arasında çelişiyor",
                    interactive=False,
                    wrap=True,
                )

            disc_btn.click(
                run_discourse,
                inputs=[disc_company, disc_topic, disc_k],
                outputs=[disc_plot, disc_info, disc_edges],
            )

    gr.Markdown(
        "---\n"
        f"**Model:** {'Phi-3.5-mini-instruct 4-bit' if USE_LLM else 'STUB (LLM yok)'} "
        f"| **NLI:** mDeBERTa-v3-base-mnli-xnli "
        f"| **Embed:** multilingual-e5-base"
    )

    submit_btn.click(
        run_query_stream,
        inputs=[question, company, year, top_k, use_nli_filter, use_isolated],
        outputs=[answer_out, metrics_out, timings_out, sources_out],
    )


if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
