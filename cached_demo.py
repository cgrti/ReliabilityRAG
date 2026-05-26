"""
ReliabilityRAG — Cached Demo Viewer (zero-GPU backup)

NEDEN:  Sunum sırasında GPU sorunu çıkarsa (VRAM dolu, CS2 açık vb.) live
        Gradio donar. Bu script GPU/LLM/NLI HİÇ yüklemez — sadece önceden
        hesaplanmış `data/demo_cache.json` dosyasını okuyup gösterir.

KULLANIM:
    1) Önce sunum öncesi: python demo_cache.py        (~5-10 dk, GPU gerekli)
    2) Sunum günü:        python cached_demo.py       (anında, GPU GEREKMEZ)
    3) UI test (cache yokken): python cached_demo.py --stub
       → Sahte ama gerçekçi içerikle UI'ı test eder. Cevaplar `[⚠️ STUB]` ile
         etiketli, demoda yanlışlıkla gösterilirse hemen anlaşılır.

UI app.py ile aynı görünümde (cevap kutusu, metrikler, timings tablosu,
kaynaklar) — sadece "Sorgula" yerine "Cevabı yükle" var. Hocaya canlıymış
gibi görünür ama her şey diskten gelir.

Eğer cache yoksa ve --stub yoksa, kullanıcıya nasıl üretileceğini söyler.
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import gradio as gr


CACHE_PATH = Path(__file__).parent / "data" / "demo_cache.json"

# Pass --stub to bypass file load and use a fabricated entry. Lets you test
# the UI shape NOW without running demo_cache.py (which needs GPU). Sahte
# içerik açıkça etiketleniyor ki yanlışlıkla sunuma sızmasın.
USE_STUB = "--stub" in sys.argv


STUB_CACHE = [
    {
        "spec": {
            "question": "[STUB] Akbank'ın karbon emisyon azaltma hedefleri nelerdir?",
            "company": "Akbank",
            "year": None,
        },
        "answer": (
            "[⚠️ STUB ÖRNEK — sahte içerik. Gerçek cevap için `python demo_cache.py` koş.]\n\n"
            "Akbank, sürdürülebilirlik raporlarında karbon nötr olma hedefini 2050 yılı "
            "olarak belirlemiştir (kaynak: 2023 raporu). Ara hedefler arasında 2030'a "
            "kadar Scope 1 ve Scope 2 emisyonlarda %42 azaltım yer almaktadır. KOBİ'lere "
            "yönelik enerji verimliliği finansman programları 2021'de başlatılmıştır."
        ),
        "retrieved_chunks": 20,
        "filtered_chunks": 19,
        "removed_chunks": 1,
        "contradiction_edges": 1,
        "timings": {
            "retrieval": 0.21,
            "isolated_answering": 0.0,
            "nli_graph_filter": 2.61,
            "final_generation": 87.34,
        },
        "total_time": 90.16,
        "chunks_detail": [
            {
                "company": "Akbank",
                "year": 2023,
                "section_type": "cevre",
                "reliability": 0.6,
                "score": 0.9123,
                "text_preview": (
                    "Akbank olarak iklim değişikliği ile mücadelede 2050 net sıfır karbon "
                    "hedefimiz doğrultusunda Scope 1 ve Scope 2 emisyonlarımızı 2030'a kadar "
                    "%42 azaltmayı taahhüt ediyoruz. Bu çerçevede tüm operasyonel binalarımızda "
                    "yenilenebilir enerji kullanımına geçiş planlanmıştır"
                ),
            },
            {
                "company": "Akbank",
                "year": 2021,
                "section_type": "strateji",
                "reliability": 0.4,
                "score": 0.8745,
                "text_preview": (
                    "KOBİ müşterilerimize yönelik düşük karbon ekonomisine geçiş için özel "
                    "finansman ürünleri geliştirilmiştir. 2021 yılında açılan enerji verimliliği "
                    "kredisi 1.2 milyar TL kullandırılmıştır"
                ),
            },
        ],
        "computed_at": "STUB (no GPU used)",
        "wall_seconds": 90.16,
    },
    {
        "spec": {
            "question": "[STUB] Kadın istihdamı ve fırsat eşitliği politikaları",
            "company": None,
            "year": None,
        },
        "answer": (
            "[⚠️ STUB ÖRNEK — sahte içerik. Gerçek cevap için `python demo_cache.py` koş.]\n\n"
            "Türk şirketleri kadın istihdamı konusunda farklı politikalar izlemektedir. "
            "Garanti BBVA 2023'te kadın yönetici oranını %38'e çıkarmıştır. Akbank fırsat "
            "eşitliği eğitimini tüm çalışanlara zorunlu kılmıştır."
        ),
        "retrieved_chunks": 20,
        "filtered_chunks": 18,
        "removed_chunks": 2,
        "contradiction_edges": 3,
        "timings": {
            "retrieval": 0.18,
            "isolated_answering": 0.0,
            "nli_graph_filter": 2.94,
            "final_generation": 102.5,
        },
        "total_time": 105.62,
        "chunks_detail": [
            {
                "company": "GarantiBBVA",
                "year": 2023,
                "section_type": "sosyal",
                "reliability": 0.5,
                "score": 0.8521,
                "text_preview": "Kadın yönetici oranımız 2023 yılında %38'e ulaşmıştır...",
            },
        ],
        "computed_at": "STUB (no GPU used)",
        "wall_seconds": 105.62,
    },
]


def load_cache():
    """Read demo_cache.json. Return [] if missing/invalid so UI can warn."""
    if USE_STUB:
        print("[cached_demo] --stub flag set, using fabricated cache (no file read)")
        return STUB_CACHE
    if not CACHE_PATH.exists():
        return []
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[cache load error] {e}")
        return []


CACHE = load_cache()
print(f"[cached_demo] loaded {len(CACHE)} cached queries from "
      f"{'STUB' if USE_STUB else CACHE_PATH}")


# ── Helpers (mirror app.py format) ──────────────────────────────────────

def _metrics_md(retrieved, kept, removed, edges, total):
    return (
        f"| Metrik | Değer |\n"
        f"|--------|------:|\n"
        f"| Getirilen (retrieved) | **{retrieved}** |\n"
        f"| Temiz kümede kalan    | **{kept}** |\n"
        f"| Çıkarılan (removed)   | **{removed}** |\n"
        f"| Çelişki kenarı        | **{edges}** |\n"
        f"| Toplam süre           | **{total:.1f}s** |\n"
    )


def _sources_md(chunks_detail):
    """
    chunks_detail: list of dicts as produced by ReliabilityRAG.query() —
    keys: company, year, section_type, reliability, score, text_preview.
    (NOT reliability_weight / text — those are the dataclass field names,
    but the pipeline serializes them differently. Verified rag_pipeline.py:222-232.)
    """
    if not chunks_detail:
        return "_(Cache içinde kaynak listesi yok)_"
    parts = []
    for i, c in enumerate(chunks_detail, 1):
        company = c.get("company", "?")
        year = c.get("year", "?")
        section = c.get("section_type", "?")
        rel = c.get("reliability", "?")
        score = c.get("score", 0.0)
        text = c.get("text_preview", "")
        parts.append(
            f"### {i}. {company} — {year}\n"
            f"**Bölüm:** `{section}` · "
            f"**Güvenilirlik:** `{rel}` · "
            f"**Benzerlik:** `{score:.3f}`\n\n"
            f"> {text}...\n"
        )
    return "\n---\n\n".join(parts)


def _timings_rows(timings: dict, total: float):
    if not timings:
        return [["TOTAL", f"{total:.2f}"]]
    rows = [[k, f"{v:.2f}"] for k, v in timings.items()]
    rows.append(["TOTAL", f"{total:.2f}"])
    return rows


# ── Question selection → fill outputs ───────────────────────────────────

def question_labels():
    """Return display labels for the dropdown (one per cached entry)."""
    if not CACHE:
        return []
    labels = []
    for i, entry in enumerate(CACHE, 1):
        spec = entry.get("spec", {})
        q = spec.get("question", "?")
        comp = spec.get("company") or "Hepsi"
        year = spec.get("year") or "Hepsi"
        labels.append(f"{i}. [{comp}/{year}] {q[:70]}")
    return labels


def load_answer(label):
    """Look up the cached entry by label and return the UI fields."""
    if not CACHE:
        msg = (
            "## ⚠️ Cache boş\n\n"
            "Önce GPU müsait bir terminalde şu komutu çalıştır:\n\n"
            "```bash\n"
            "python demo_cache.py\n"
            "```\n\n"
            "Bu, `data/demo_cache.json` dosyasını üretir (~5-10 dk).\n"
            "Sonra `cached_demo.py`'i yeniden başlat."
        )
        return msg, "", [], ""

    if not label:
        return "(Soldan bir soru seç)", "", [], ""

    # Match by index prefix "1. ..." / "2. ..."
    try:
        idx = int(label.split(".", 1)[0]) - 1
    except Exception:
        return f"[Hatalı label: {label}]", "", [], ""
    if not (0 <= idx < len(CACHE)):
        return f"[Index out of range: {idx}]", "", [], ""

    entry = CACHE[idx]
    answer = entry.get("answer", "(Cevap yok)")
    metrics = _metrics_md(
        entry.get("retrieved_chunks", 0),
        entry.get("filtered_chunks", 0),
        entry.get("removed_chunks", 0),
        entry.get("contradiction_edges", 0),
        entry.get("total_time", 0.0),
    )
    timings = _timings_rows(entry.get("timings", {}), entry.get("total_time", 0.0))
    sources = _sources_md(entry.get("chunks_detail", []))

    # Append cache metadata to answer
    computed = entry.get("computed_at", "?")
    wall = entry.get("wall_seconds", entry.get("total_time", 0.0))
    answer_with_meta = (
        f"{answer}\n\n"
        f"---\n"
        f"_📦 Cache: {computed} · ⏱ Gerçek hesaplama süresi: {wall:.1f}s_"
    )
    return answer_with_meta, metrics, timings, sources


# ── UI ──────────────────────────────────────────────────────────────────

DESCRIPTION = """
# ReliabilityRAG — Önbelleğe Alınmış Demo (Cache Viewer)

**248 Türk kurumsal entegre raporu** (63 şirket, 2015-2024, 182K chunk) üzerinde
çalışan, **GAT tabanlı dinamik filtreleme** ile bilgi çelişkilerini otomatik tespit
eden RAG sistemi.

**6 aşamalı pipeline:** Retrieval → (Isolated Answering) → NLI Contradiction Graph
→ MWIS Filtering → GAT Dynamic Filtering → Final Generation

> ⚡ **Not:** Bu görünümdeki cevaplar **önceden hesaplandı** (hesaplama süresi her
> sorunun yanında gösteriliyor). Aynı sorgu live `app.py` ile de koşturulabilir;
> bu cache demo süresinde GPU dalgalanmalarına karşı yedektir.

_CSE496 Bitirme Projesi — Danışman: Prof. Yusuf Sinan Akgül_
"""


with gr.Blocks(title="ReliabilityRAG (Cached)") as app:
    gr.Markdown(DESCRIPTION)

    if not CACHE:
        gr.Markdown(
            "## ⚠️ Cache henüz oluşturulmamış\n\n"
            "Bu viewer `data/demo_cache.json` dosyasını okur. Önce şu komutu koş:\n\n"
            "```bash\npython demo_cache.py\n```\n\n"
            "(GPU gerektirir, ~5-10 dakika sürer. Sonra bu uygulamayı yeniden başlat.)"
        )

    with gr.Row():
        with gr.Column(scale=2):
            question_dropdown = gr.Dropdown(
                choices=question_labels(),
                label="Önbellekteki sorular",
                value=question_labels()[0] if CACHE else None,
                interactive=True,
            )
            load_btn = gr.Button("📥 Cevabı yükle", variant="primary", size="lg")
            gr.Markdown(
                f"**Toplam önbellekli sorgu:** {len(CACHE)}  \n"
                f"**Cache dosyası:** `{CACHE_PATH.name}`"
            )

        with gr.Column(scale=3):
            answer_out = gr.Textbox(label="📝 Cevap", lines=14)
            metrics_out = gr.Markdown(label="📊 Pipeline metrikleri")
            timings_out = gr.Dataframe(
                headers=["Aşama", "Süre (s)"],
                label="⏱ Aşama süreleri (gerçek hesaplama)",
                datatype=["str", "str"],
                interactive=False,
            )

    with gr.Accordion("📚 Kaynaklar (filtreleme sonrası temiz küme)", open=False):
        sources_out = gr.Markdown()

    gr.Markdown(
        "---\n"
        "**Mod:** CACHED VIEWER (LLM/NLI/embedder yüklü değil — diskten okuyor) "
        "| **Live mod için:** `python app.py`"
    )

    load_btn.click(
        load_answer,
        inputs=[question_dropdown],
        outputs=[answer_out, metrics_out, timings_out, sources_out],
    )
    # Auto-load on dropdown change too
    question_dropdown.change(
        load_answer,
        inputs=[question_dropdown],
        outputs=[answer_out, metrics_out, timings_out, sources_out],
    )


if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7861,  # different from live app.py (7860) — can run both
        inbrowser=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
