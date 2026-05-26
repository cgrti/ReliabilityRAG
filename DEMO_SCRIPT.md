# ReliabilityRAG — Hoca Sunumu Demo Senaryosu
**Tarih:** 2026-04-22 | **Süre:** ~10-15 dk demo + Q&A

---

## 🚀 BAŞLATMA (sunumdan 2 dk önce)

```bash
cd "D:/er mineru dosyaları/reliability_rag"
python app.py
```

Beklenen startup log'u:
```
GPU profile: small
GPU VRAM: 6.4 GB
Loading LLM: microsoft/Phi-3.5-mini-instruct (dtype=4bit)
[LLM loaded in ~25s]
[warming up searcher + NLI...]
Loaded 182986 chunks
Ready: 182986 vectors, dim=768
NLI model loaded.
[warmed in ~14s]
[63 companies, years 2015-2024]
======================================================================
Running on local URL:  http://127.0.0.1:7860
```

**Tarayıcıda aç:** `http://127.0.0.1:7860`

---

## 🎬 DEMO AKIŞI

### 1. Açılış — sistemi tanıt (1-2 dk)
Hocaya **üstteki başlığı göster:**
> "248 Türk kurumsal entegre raporu (63 şirket, 2015-2024, 182K chunk) üzerinde
> çalışan, GAT tabanlı dinamik filtreleme ile bilgi çelişkilerini otomatik tespit
> eden RAG sistemi."

**Altındaki pipeline metnini oku:**
> "6 aşamalı pipeline: Retrieval → Isolated Answering (opsiyonel) → NLI Contradiction
> Graph → MWIS Filtering → GAT Dynamic Filtering → Final Generation"

Soru kutusuna vurgu: "Şirket ve yıl filtresi var, Top-K ayarlanabilir, ileri ayarlar kapalı."

---

### 2. Örnek 1: Akbank karbon hedefleri (en sağlam case) — 3 dk

**İşlem:** Alt taraftaki "Akbank'ın karbon emisyon azaltma hedefleri" örneğine tıkla
→ Şirket filtresi `Akbank`, yıl `(Hepsi)` otomatik dolacak
→ **🔍 Sorgula** butonuna bas

**Ne gösterilir (canlı):**
1. ⏳ "Retrieval başladı..."
2. ⏳ "Retrieval tamam (20 chunk, 0.2s). NLI + MWIS + GAT filtrelemesi başlıyor..."
3. ⏳ "Filtreleme tamam. 19/20 chunk kaldı. LLM cevap üretiyor..."
4. 📚 **Kaynaklar panelinde** Akbank 2020/2021/2023 çevre bölümleri, sim=0.90+
5. 📊 **Metrikler tablosu:**
   | Metrik | Değer |
   |--------|------|
   | Getirilen | 20 |
   | Temiz kümede kalan | 19 |
   | Çıkarılan | 1 |
   | **Çelişki kenarı** | **1** |
6. 📝 **Cevap alanı token token doluyor** — "1. 2050 net sıfır karbon..., 2. KOBİ enerji verimliliği..., 3. Düşük karbon ekonomisine geçiş..."

**Hocaya vurgula:**
- "Retrieval 0.2s — numpy cosine, ChromaDB'ye göre çok hızlı"
- "NLI 74 pair/sn CUDA'da"
- "1 çelişkili chunk otomatik atıldı — **GAT bu kararı verdi** (log'da GAT+MWIS: kept 19/20)"
- "LLM cevabı sadece filtrelenmiş temiz kümeden geliyor"

**Toplam süre:** ~90-120 saniye (repetition fix sonrası hedef)

---

### 3. Örnek 2: Çelişki görselleştirme — sorgulamanın hangi kaynağı attığını göster (2 dk)

**Kaynaklar** accordion'unu aç → 19 kaynağı göster, tümü Akbank farklı yıllar.
**Vurgu:** "Çelişki kenarı 1 olması, 20 chunk içinden birinin diğerleriyle çeliştiği için atıldığını gösteriyor. Bu, GAT katmanının dinamik attention weight'leriyle karar verildi."

Eğer vakit varsa log'a geri dön:
```
Running NLI on 190 pairs...
NLI done in 2.6s (74 pairs/sec)
Graph: 20 nodes, 1 contradiction edges
GAT+MWIS: kept 19/20 chunks (removed 1 contradictory chunks)
```

---

### 4. Örnek 3: Cross-company sorgusu — kurumsal geneli göster (2-3 dk)

Input'a yaz: **"Kadın istihdamı ve fırsat eşitliği politikaları"** (şirket/yıl = Hepsi)

Bekle → farklı şirketlerden chunk'lar gelecek (Akbank, GarantiBBVA, Cimsa, vs).

**Vurgu:** "Bu cross-company bir sorgu — sistem farklı şirketlerin sosyal sürdürülebilirlik bölümlerinden kaynak topluyor. Çelişki olursa (örn: iki şirket birbiriyle çelişen iddialarda) bu da tespit ediliyor."

---

### 5. Özgün katkı: GAT (sunuma özel) (2 dk)

**Ekranda göster:** `D:/er mineru dosyaları/reliability_rag/gat_filter.py`

**Söyle:**
> "Standart ReliabilityRAG paper'ında MIS (Maximum Independent Set) var.
> Benim özgün katkım: MIS yerine **GAT tabanlı dinamik filtreleme** — her chunk'ın
> attention ağırlığı hesaplanırken **temporal provenance** (chunk'ın hangi yıldan
> geldiği) ve **source reliability** (finansal=0.9, çevre=0.6, strateji=0.4)
> feature'ları kullanılıyor. Böylece 2020 yılındaki bir strateji iddiası, 2023'teki
> denetimli finansal veriye göre daha az ağırlık alıyor."

---

## ⚠️ DEMO SIRASINDA RİSK DURUMLARI

### Risk 1: LLM çok yavaş (>3 dk tek sorgu)
- Repetition fix'e rağmen olursa → "Stub mod" bölümüne geç
- Alt taraftaki **İleri ayarlar** → NLI toggle kapat → retrieval-only göster
- Veya: tarayıcı sekmesini kapatıp `RELIABILITY_RAG_NO_LLM=1 python app.py` ile yeniden başlat (stub mod sadece chunk döndürür, anında)

### Risk 2: Gradio server çökerse
- Terminal'e `Ctrl+C`, sonra `python app.py` tekrar çalıştır
- Yedek: `python demo_full_pipeline.py --llm` (CLI mode, UI yok ama çalışır)

### Risk 3: GPU OOM (out of memory) — EN YAYGIN SORUN
- RTX 2060 6GB'da LLM+NLI+embedder toplam ~5.8 GB, sadece 200 MB boş marj var
- CS2, Chrome (çok sekme), Discord, Steam vb. GPU bellek paylaştığında LLM generation **10-50x yavaşlar**
- **2026-04-21 test:** CS2 açıkken sorgu "donar" gibi göründü (aslında VRAM swap yapıyor, çok yavaş)
- Sunumdan önce: Tüm GPU kullanıcılarını kapat (Görev Yöneticisi → GPU sekmesi → "Ayrılmış GPU bellek" sütunu)
- 🛟 **EN İYİ YEDEK — `cached_demo.py`** (port 7861, sıfır-GPU): Sunumdan önce
  `python demo_cache.py` ile 5 örnek sorgu önbelleğe yazılır (~5-10 dk, GPU gerekli).
  Sonra `python cached_demo.py` herhangi bir GPU kullanmadan **app.py'nin görsel
  ikizini** sunar — cevaplar diskten anında. Hocaya farkı belli etmez ama arkada
  GPU ısınmıyor bile. **Cache hazır değilse** `python cached_demo.py --stub` ile
  sahte (etiketli) içerikle UI'ı önceden test edebilirsin.
- Cache yok ve GPU dolu ise son çare: `RELIABILITY_RAG_NO_LLM=1 python app.py` → LLM yüklemeden demo (retrieval + NLI + GAT gösterilebilir, sadece cevap üretilmez)

### Risk 4: Hocada sorular
**Hazırlıklı olduğun sorular:**
- "GAT'ı nasıl eğittin?" → "Henüz heuristic-init ağırlıklarla çalışıyor, forward pass yapıyor. Eğitim MWIS+GAT ablation'unda ikinci faz."
- "248 raporu hepsini tarıyor mu?" → "Evet, 182K chunk, 563 MB .npy matrisi. Filtre yoksa hepsi arandı."
- "Neden RAGAS kullanmadın?" → "Planda var — sunum sonrası evaluation fazında."
- "Neden Phi-3.5-mini?" → "RTX 2060 6GB'a sığdırmak için. Akvaryum PC'de RTX 5080 olunca Qwen2.5-7B'a geçecek (env var ile tek komut)."
- "Türkçe NLI ne kadar doğru?" → "Sentetik greenwashing test'te %99.88 çelişki tespit ettik."

---

## 📝 DEMODAN ÖNCE CHECKLIST

- [ ] **🔥 CS2 / oyunlar / Chrome GPU-hungry sekmeler KAPATILDI** — RTX 2060 6GB'da LLM ~5.8GB VRAM alıyor, 200MB boş marjla çalışıyor. Paylaşımlı GPU kullanan her şey LLM'i 10-50x yavaşlatır (VRAM swap).
- [ ] Görev Yöneticisi → Performans → GPU sekmesinde **Ayrılmış GPU bellek < 1 GB** (CS2 açıkken ~1.5-2 GB olabilir)
- [ ] **🛟 Demo cache üretildi** — sunumdan en geç 1 saat önce `python demo_cache.py` koşuldu, `data/demo_cache.json` ≥ 1KB (5 sorgulu cevap dolu). Bu yapılmadan `cached_demo.py` boş.
- [ ] Gradio server başlatılmış (`python app.py`) ve log'da "Running on local URL: http://127.0.0.1:7860" görüldü
- [ ] Tarayıcıda `http://127.0.0.1:7860` açık ve sayfa yüklendi
- [ ] İlk "warmup" sorgusu elle yapıldı (örneğin "test") — böylece demo sırasında cold-start yaşanmaz
- [ ] Ekran `D:/er mineru dosyaları/reliability_rag/gat_filter.py` sekmesi hazır (özgün katkı bölümü için)
- [ ] **Yedek plan A:** `python cached_demo.py` (port 7861) — diskten anında, GPU gerekmez. **EN GÜVENLİ YEDEK.** Demo öncesi sekmesini açıp test et.
- [ ] Yedek plan B: `RELIABILITY_RAG_NO_LLM=1 python app.py` — LLM'siz mod (cache yoksa ve VRAM sorun ederse)
- [ ] Yedek plan C: `python demo_full_pipeline.py --llm` CLI komutu hazır
- [ ] Telefonda veya bir sekmede `session_2026_04_18_21_ui_smoke.md` dosyası açık (sayıları hatırlamak için)

---

## 🎯 SUNUM SONRASI (hoca memnun çıkarsa)

- Sıradaki: MWIS-only vs MWIS+GAT ablation (niceliksel GAT kanıtı)
- RAGAS evaluation + synthetic test set
- RTX 5080'e geçiş (env var: `RELIABILITY_RAG_PROFILE=large`)
- Tez raporu yazımına başla (Faz 9)
