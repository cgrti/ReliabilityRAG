"""Weekly progress report PDF generator — 1 page, Turkish."""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ── Register a Turkish-capable font (Arial fallback to Calibri) ──────────
WIN_FONTS = [
    ("Arial",      r"C:\Windows\Fonts\arial.ttf"),
    ("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"),
    ("Arial-Italic", r"C:\Windows\Fonts\ariali.ttf"),
]
for name, path in WIN_FONTS:
    if Path(path).exists():
        pdfmetrics.registerFont(TTFont(name, path))
FONT_BODY = "Arial" if Path(WIN_FONTS[0][1]).exists() else "Helvetica"
FONT_BOLD = "Arial-Bold" if Path(WIN_FONTS[1][1]).exists() else "Helvetica-Bold"


# ── Styles ──────────────────────────────────────────────────────────────
STY = getSampleStyleSheet()

H1 = ParagraphStyle(
    "H1", parent=STY["Heading1"],
    fontName=FONT_BOLD, fontSize=13, leading=15,
    spaceBefore=0, spaceAfter=4, textColor="#1f3864",
)
META = ParagraphStyle(
    "Meta", parent=STY["Normal"],
    fontName=FONT_BODY, fontSize=8.5, leading=10.5, textColor="#444",
    spaceAfter=6,
)
H2 = ParagraphStyle(
    "H2", parent=STY["Heading2"],
    fontName=FONT_BOLD, fontSize=10, leading=12,
    spaceBefore=8, spaceAfter=2, textColor="#1f3864",
)
BODY = ParagraphStyle(
    "Body", parent=STY["Normal"],
    fontName=FONT_BODY, fontSize=9, leading=11.4,
    alignment=TA_JUSTIFY, spaceAfter=2,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY,
    fontSize=9, leading=11.2, leftIndent=10, bulletIndent=2,
)


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(t, BULLET), leftIndent=10, value="•") for t in items],
        bulletType="bullet", start="•", leftIndent=10, bulletFontName=FONT_BODY,
        bulletFontSize=9,
    )


# ── Document content ────────────────────────────────────────────────────
OUT = Path(__file__).parent / "data" / "weekly_report_2026-05-12.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=1.6*cm, rightMargin=1.6*cm,
    topMargin=1.3*cm, bottomMargin=1.3*cm,
    title="Haftalık İlerleme Raporu — 12 Mayıs",
    author="Çağrı Tirelioğlu",
)

story = []

story.append(Paragraph(
    "ReliabilityRAG — Haftalık İlerleme Raporu (12 Mayıs 2026)", H1))
story.append(Paragraph(
    "<b>Öğrenci:</b> Çağrı Tirelioğlu &nbsp;·&nbsp; "
    "<b>Ders:</b> CSE496 Bitirme Projesi &nbsp;·&nbsp; "
    "<b>Danışman:</b> Prof. Dr. Yusuf Sinan Akgül<br/>"
    "<b>Proje:</b> Automated Knowledge Conflict and Inconsistency Detection "
    "via Graph-Based Reliable RAG in Multi-Document Systems",
    META,
))

# 1) Bu hafta yapılanlar
story.append(Paragraph("1. Bu hafta yapılanlar (5–12 Mayıs)", H2))
story.append(bullets([
    "<b>Ablation altyapısı kuruldu</b> (<i>ablation_mwis_vs_gat.py</i>): MWIS vs MWIS+GAT "
    "karşılaştırması, JSON çıktı ve markdown rapor üretimi.",
    "<b>GAT supervised training pipeline'ı yazıldı</b> (<i>train_gat.py</i>): BCE + "
    "contrastive margin loss, Leave-One-Out CV, early stopping, weight decay, gradient "
    "clip. Üç hyperparametre rejimi sistematik test edildi: <i>collapse</i> (lr=5e-3, "
    "blend=1.0), <i>plateau</i> (lr=5e-3, blend=0.7), <i>stable</i> (lr=1e-3, blend=0.5).",
    "<b>5 ciddi GAT bug'ı bulundu ve düzeltildi:</b> scorer'ın son lineer katmanı "
    "rastgele başlatılıyordu; ReLU işaret hatası kontradiksiyon sinyalini öldürüyordu; "
    "GAT attention aggregation çelişki edge'iyle bağlı node skorlarını eşitliyordu — "
    "<i>skip-connection</i> (statik+GAT blend) ile çözüldü.",
    "<b>NLI çelişki tespiti 5 round kalibre edildi</b>: threshold 0.5 → 0.35, hibrit "
    "<i>max(nli, num_conflict)</i>, min-ratio sayısal karşılaştırma (chunk-içi false "
    "positive temizlendi), Türkçe binlik ayraçlı sayı parser'ı eklendi. <b>Filtreleme "
    "recall %33 → %67</b>, temiz chunk korunması <b>%100 → %88</b>.",
    "<b>Sentetik test seti 9 → 19 case'e genişletildi</b>: 6 boyut "
    "(<i>temporal · scope · interdepartmental · gat_discriminating · cross_company · "
    "zero_claim</i>). Son iki boyut bu hafta yazıldı.",
    "<b>Söylem Değişimi Görselleştirmesi</b> (Saliha Hocam'ın 10 Mayıs isteği): "
    "Plotly tabanlı interaktif timeline — şirket × konu seçince yıllar boyunca "
    "chunk benzerliği, NLI çelişki edge'leri kırmızı kesik çizgi ile çiziliyor. "
    "Gradio arayüzüne <i>📈 Söylem Timeline</i> sekmesi ve <i>En Çelişkili Çiftler</i> "
    "tablosu eklendi.",
    "<b>Tezsel sonuçlar raporu</b> (<i>thesis_report.md</i> + LaTeX tablo "
    "<i>thesis_report.tex</i>) ile <b>plato analizi belgesi</b> "
    "(<i>plato_analysis.md</i>) hazırlandı.",
]))

# 2) Niceliksel sonuçlar
story.append(Paragraph("2. İlerleme — Niceliksel sonuçlar", H2))
story.append(bullets([
    "19-case ablation: <b>MWIS 11/19 = trained GAT 11/19</b> — regresyon yok, "
    "<i>gat_only_ok=0</i>, <i>mwis_only_ok=0</i>.",
    "Boyut bazında: <i>interdepartmental</i> <b>3/3 (%100)</b>, <i>cross_company</i> "
    "<b>2/3</b> (yeni boyut), <i>scope</i> 2/3, <i>gat_discriminating</i> 2/5.",
    "Pipeline gecikmesi: ortalama 0.08–0.17 s/sorgu (hedef &lt; 5 s).",
    "Söylem timeline örnekleri: GarantiBBVA × kadın çalışan oranı 190 çelişki "
    "edge'i (2017–2024); NuhCimento × yenilenebilir enerji 200 edge.",
]))

# 3) Takıldığım yerler
story.append(Paragraph("3. Takıldığım yerler", H2))
story.append(bullets([
    "<b>GAT plato</b>: eğitilmiş GAT mevcut 19-case'de MWIS ile eşit. Modelin "
    "training set'i bile ezberleyemediği gözlendiğinden bu <i>overfit değil, "
    "underfit + dataset diversity bottleneck</i> olarak yorumlandı (üç hyperparametre "
    "rejiminde sistematik gözlem).",
    "<b>NLI kısa devresi</b>: 3–4 testte NLI hiç çelişki edge'i çizmiyor — özellikle "
    "Türkçe <i>revize edildi</i> tipi temporal güncelleme cümleleri ve <i>sıfır atık</i> "
    "tipinde nicel-olmayan greenwashing. Tavanı 11/19 + 4 ≈ 15/19'a sınırlıyor.",
    "<b>Donanım kısıtı</b>: RTX 2060 6 GB üzerinde Phi-3.5-mini 4-bit + mDeBERTa-base + "
    "e5-base toplamı ~5.8 GB; geniş datasette GAT re-training ve full evaluation lokal "
    "sığmıyor. <b>12 Mayıs itibarıyla akvaryum laboratuvarındaki RTX 5080'e erişim "
    "Saliha Hocam aracılığıyla sağlandı</b>.",
]))

# 4) Önümüzdeki hafta
story.append(Paragraph("4. Önümüzdeki hafta planı (12–19 Mayıs)", H2))
story.append(bullets([
    "Akvaryum RTX 5080'e geçiş: <i>RELIABILITY_RAG_PROFILE=large</i> profilinin "
    "devreye alınması (Qwen2.5-7B + mDeBERTa-large + batch 128).",
    "Sentetik test setinin 19 → 30+ case'e çıkarılması; ardından geniş profille GAT "
    "re-training ve ablation tekrarı.",
    "NLI fine-tuning veri hazırlığı: 248 entegre rapordan 100+ manuel etiketli "
    "Türkçe çelişki/destek pair'i — özellikle <i>revize-tipi</i> temporal kalıplar.",
    "Phi-3.5-mini judge tabanlı minimal faithfulness scorer prototipi (RAGAS "
    "yerine yerel alternatif).",
    "Tez raporunun yazımına başlanması: <i>thesis_report.md</i> şablonu temel "
    "alınarak bölümlerin genişletilmesi.",
]))

doc.build(story)
print(f"[saved] {OUT}")
print(f"        {OUT.stat().st_size/1024:.1f} KB")
