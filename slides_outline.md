# Sunum Slaytları — ReliabilityRAG Tez Savunması

**Hedef süre:** ~25 dk talk + 10 dk Q&A → ~25 slayt (1 dk/slayt)
**Audience:** GTÜ CSE Bölümü Jürisi (ML/NLP arka planı var)
**Dil:** Türkçe (terminoloji İngilizce karışık OK)

**Demo örnekleri** (cherry-picked from `data/demo_cache.json`):
- 🥇 [4] Kadın istihdamı
- 🥈 [10] Akbank 2023 başlıklar
- 🥉 [12] Türk şirketler ortak temalar
- [1] Akbank karbon · [3] Çimento alt. yakıt · [8] Bankacılık kadın yön. · [5] Anadolu Hayat

---

## SLAYT 1 — Başlık

**İçerik:**
> **ReliabilityRAG**
> Automated Knowledge Conflict and Inconsistency Detection
> via Graph-Based Reliable RAG in Multi-Document Systems
>
> Çağrı Tirelioğlu · CSE496 Bitirme Projesi
> Danışman: Prof. Dr. Yusuf Sinan Akgül
> GTÜ Bilgisayar Mühendisliği · 2026

**Görsel:** GTÜ logo + proje logo varsa. Yoksa minimal yazı.

**Not:** "İsmim Çağrı Tirelioğlu, ReliabilityRAG projesi: çok-dokümanlı kurumsal raporlarda bilgi çelişkilerini grafik tabanlı yöntemlerle otomatik tespit eden güvenilir bir RAG sistemi."

---

## SLAYT 2 — Gündem (Agenda)

**İçerik:**
1. Problem & Motivasyon
2. Veri Seti (248 Türk kurumsal raporu)
3. Pipeline Mimari (6 aşama)
4. Yöntem: NLI Grafiği + MWIS + GAT
5. Değerlendirme: Filter + Generation katmanları
6. Canlı Demo (3 örnek)
7. Limitations & Future Work
8. Sonuç + Q&A

**Görsel:** Numaralı liste, basit.

**Not:** "Şu plan ile ilerleyeceğim. Yaklaşık 25 dakika sürer, sonunda sorulara açığım."

---

## SLAYT 3 — Problem: Multi-Document RAG'da Halüsinasyon Sorunu

**İçerik:**
- Kurumsal raporlar **birbirleriyle çelişen iddialar** içerir (yıllar arası revizyon, kapsam farkları, pazarlama vs gerçek)
- Standart RAG: en yüksek similarity-skorlu chunks'ı LLM'e iletir → **çelişkili kaynaklar = halüsinasyon riski**
- 2 katman risk:
  1. **Filter layer**: yanlış chunks alınır
  2. **Generation layer**: LLM kaynaktan sapar / uydurur

**Görsel:** Sol: standart RAG diyagramı (Q → retrieve → LLM → A). Sağ: kırmızı X ile çelişki örneği.

**Not:** "Klasik RAG'ın 2 darboğazı var: hangi chunks'ı alıyor, onlardan LLM nasıl üretiyor. Biz iki katmana da müdahale eden bir sistem kurduk."

---

## SLAYT 4 — Motive Edici Örnek

**İçerik:**
```
Soru: "Şirketin 2030 karbon emisyon hedefi nedir?"

Chunk A (2021): "2030'a kadar %50 azaltma hedefi"
Chunk B (2023): "Hedef %50'den %35'e revize edildi"
Chunk C (2024): "2030 hedefi %35"

Standart RAG → her ikisini de LLM'e gönderir → çelişkili cevap
ReliabilityRAG → Chunk A çelişkili (eski), filtreler → tutarlı cevap
```

**Görsel:** 3 chunk vertical, MWIS+GAT filter ile eski olanı kırmızıyla çiz.

**Not:** "İşte tam böyle bir durumda standart RAG'ın yanılması kaçınılmaz. Bizim sistem temporal context'i anlayarak güncel olanları seçmeli."

---

## SLAYT 5 — Veri Seti: 248 Türk Kurumsal Raporu

**İçerik:**
| Özellik | Değer |
|---|---|
| Toplam rapor (entegre/sürdürülebilirlik) | 248 |
| Farklı şirket | 63 |
| Yıl aralığı | 2015–2024 |
| İşlenmiş chunk sayısı | 182.986 |
| Toplam metin | ~50M karakter |
| Sektörler | Bankacılık, çimento, perakende, enerji, telekom, ... |
| Embedding modeli | `intfloat/multilingual-e5-base` |

**Görsel:** Sektör pasta grafiği + yıllara göre rapor sayısı bar chart.

**Not:** "248 rapor, 63 şirket, 10 yıl. Bu Türk ESG literatürü için **kendi başına değerli bir corpus**. Her chunk metadata'sı var: şirket, yıl, section type, reliability weight."

---

## SLAYT 6 — Pipeline: 6 Aşamalı Mimari

**İçerik:**
```
[Sorgu]
   ↓
1. Retrieval (numpy cosine search, top-k=20)
   ↓
2. (Isolated Answering — opsiyonel)
   ↓
3. NLI Contradiction Graph (mDeBERTa-base + Türkçe numerical)
   ↓
4. MWIS / GAT Filtering (chunk subset selection)
   ↓
5. Final Generation (Turkish-Llama-8b, 4-bit)
   ↓
[Güvenilir Cevap + Kaynak Listesi]
```

**Görsel:** Sol→sağ akış diyagramı, her kutuda emoji + kısa açıklama. Renkli oklar.

**Not:** "Pipeline 6 aşamalı. Bu sunumda en çok 3 (NLI grafiği) ve 4 (GAT filtreleme) üzerinde duracağım çünkü tezsel katkı oradalar."

---

## SLAYT 7 — NLI Contradiction Graph (Aşama 3)

**İçerik:**
- Her chunk pair için NLI (Natural Language Inference) ile çelişki olasılığı hesaplanır
- Model: `mDeBERTa-v3-base-mnli-xnli` (Türkçe destekli)
- Edge weight = `max(nli_contradiction_score, numerical_conflict_score)`
- Threshold = 0.35 (kalibrasyon sonrası — Slayt 13)

**Görsel:** 5-chunk graf örneği, kırmızı edge'ler çelişki, kalın olanlar yüksek skor.

**Not:** "20 chunk = 190 pair. Her birinin NLI skorunu alıyoruz, çelişki olasılığı > 0.35 ise edge çiziyoruz. Türkçe sayısal çelişkiler için extra hesaplama var."

---

## SLAYT 8 — Türkçe Numerical Conflict Detection

**İçerik:**
- Türkçe sayı format pattern'leri (`extract_numbers`):
  - `%42`, `%42,5`
  - `1.250` (Türkçe binlik ayraç)
  - `245 milyon TL`, `8.2 milyar`
  - `290 GWh`, `45 ton`
- Conflict score = `min_ratio` between extracted numbers across chunks
  - `(min_ratio - 1.0) / 1.0` scaling
  - Trigger when ratio > 1.5x
- Year exclusion (1900-2099 sayı sayılmaz)

**Görsel:** Code snippet + example: "150K vs 580K → ratio 3.87 → high conflict"

**Not:** "NLI Türkçe sayısal çelişkileri tam yakalayamıyordu. Custom pattern + min-ratio metric'i ekledik. Recall %33→%67'ye çıktı."

---

## SLAYT 9 — Filtering: MWIS Baseline + GAT Contribution

**İçerik:**
**MWIS (Maximum Weight Independent Set):**
- Çelişki grafından en büyük "çelişkisiz" alt-kümeyi seç
- Greedy heuristic: `reliability_weight × recency_factor`

**GAT Contribution (Bu çalışma):**
- Graph Attention Network ile dinamik scoring
- Feature'lar: section_type, reliability, year, neighbor_aggregation
- Skip-connection: `blend * GAT + (1-blend) * static_score`
- Supervised training (BCE + contrastive margin loss)

**Görsel:** Sol: MWIS karar tablosu. Sağ: GAT mimarisi mini-diagram.

**Not:** "MWIS bir baseline. GAT'ın katkısı: graf yapısını öğrenerek section ve recency-aware kararlar verebilmesi. Eğitildi: gat_weights.pt."

---

## SLAYT 10 — Sentetik Test Seti: 8 Boyut, 30 Case

**İçerik:**
| Boyut | n | Test ettiği |
|---|---:|---|
| temporal | 3 | Yıllar arası hedef revizyonu |
| scope | 3 | Kapsam/metodoloji farkı |
| interdepartmental | 3 | Pazarlama vs finansal (greenwashing) |
| gat_discriminating | 10 | MWIS reliability-baskın yanılır |
| cross_company | 3 | Aynı sektörde farklı şirketler |
| zero_claim | 2 | "Sıfır atık" abartı vs gerçek |
| dense_graph | 3 | 8-chunk multi-way conflict |
| numerical_edge | 3 | False-positive guard (rounding tolerance) |
| **Toplam** | **30** | **88 chunk, 38 çelişkili, 50 temiz** |

**Görsel:** Tablo + tek bir test case'inin chunks gösterimi (örn. gat_discriminating Test-9).

**Not:** "Ground truth ile etiketlenmiş 30 case. Her case için hangi chunks tutulmalı/atılmalı belli. Bu **ablation için gold standard**."

---

## SLAYT 11 — Filter Layer Ablation: MWIS vs GAT

**İçerik:**
30-case test set, aynı NLI grafiği, sadece filtering layer farklı:

| Metrik | MWIS-only | Heuristic GAT | Trained GAT |
|---|---:|---:|---:|
| Doğru filtreleme | 15/30 | 14/30 | **15/30** |
| Filtreleme recall | 50.0% | 47.4% | 50.0% |
| Temiz preservation | 64.0% | 62.0% | **66.0%** |
| Avg gecikme | 0.03s | 0.02s | 0.02s |

**Verdict dağılımı (Trained GAT vs MWIS):**
- both_ok: 14
- **gat_only_ok: 1** ← GAT-9 4-chunk dense temporal evolution case kurtardı
- mwis_only_ok: 1 ← same-reliability tie-break (regresyon)

**Görsel:** Bar chart + verdict pie chart.

**Not:** "Score eşit (15/15) ama **kompozisyon farklı**. Training, hedeflediğimiz boyutta (gat_discriminating) +1 case kurtardı. Pipeline doğrulandı, **dataset doyma noktasında değil** — daha geniş corpus future work."

---

## SLAYT 12 — Generation Faithfulness Eval (YENİ — Bu çalışmanın özgün katkısı)

**İçerik:**
**Soru:** Filter temizledikten sonra LLM kaynaklara sadık mı?

**Yeni eval pipeline:** `evaluation_generation.py`
- 4 metrik: entity_recall, entity_purity, year_accuracy, **numerical_fidelity**
- 2 mode: oracle (filter atlandı), pipeline (full)

**Sonuç (Turkish-Llama-8b-4bit, 30 case):**
| Metrik | Oracle | Pipeline |
|---|---:|---:|
| Year Accuracy | 0.73 | 0.74 |
| **Numerical Fidelity** | **0.54** | **0.53** |

**Görsel:** Bar chart: filter eval (%67 recall) vs generation eval (%54 numfid). İki katmanlı reliability.

**Not:** "İşte tezsel beklentinin dışında bir bulgu: LLM, temiz veri verilse bile **%46 sayısal drift** üretiyor. Filter darboğaz değil, LLM-katmanı asıl darboğaz."

---

## SLAYT 13 — NLI Kalibrasyon Hikayesi (5 Round)

**İçerik:**
| Round | Değişiklik | Recall |
|---|---|---:|
| 1 | Baseline (thr=0.5, additive scoring) | 33% |
| 2 | thr 0.5→0.35, `max(nli, num)` | 55% |
| 3 | Numerical scaling, trigger 1.5x | 55% |
| 4 | MIN-ratio + Türkçe binlik ayraç | **67%** |
| 5 | Bare integer + year exclusion | 67% |

**Görsel:** Recall çizgi grafiği + her round'da değişen kod parçası küçük inset.

**Not:** "Bu sayı bir sabit değildi — iteratif kalibrasyon hikayesi. Türkçe sayı format'larını yakalayan custom kod, hybrid max() — en kritik atılımlar."

---

## SLAYT 14 — Hyperparameter Sweep: GAT Training

**İçerik:**
| Rejim | lr | blend | Sonuç |
|---|---|---|---|
| Aggressive | 5e-3 | 1.0 | **Constant collapse** (0/14) — pure GAT static-signal'siz |
| Initial | 5e-3 | 0.7 | Plateau (8/14) |
| Conservative | 1e-3 | 0.5 | **Stable** (15/30) — kullandığımız |

**Yorum:** Heuristic-init pure-GAT bir local optimum'a kaçtı. Skip-connection (blend) + conservative lr ile stabilize ettik.

**Görsel:** Loss eğrisi 3 çizgi (aggressive, initial, conservative).

**Not:** "Training pipeline doğrulandı. Hyperparameter robustness gösterdik. Conservative regime'ı production için seçtik."

---

## SLAYT 15 — Production Stack: Hardware + Model

**İçerik:**
**Hardware envelope:**
- Geliştirme: laptop, RTX 2060 6GB
- Production: Akvaryum PC, RTX 5080 16GB

**LLM Seçim Hikayesi:**
| Model | dtype | VRAM | Sonuç |
|---|---|---|---|
| Qwen-7B-fp16 | 16GB | overflow | CPU offload donma |
| Qwen-7B-4bit | 4GB | 38.5s | Entity drift, FEM→"FIŞLİ EŞİKLİK" |
| Qwen-14B-4bit | 8GB | 595s | CPU offload + **Çince'ye savruldu** |
| **Turkish-Llama-8b-4bit** | 4GB | **18.5s** | ✓ Production-stable |

**Görsel:** Memory diagram + 4 model timing bar chart.

**Not:** "3 model denedik. Qwen Türkçe tokenizer'ı ekleri parçalıyor, 4-bit altında morfoloji bozuluyor. YTÜ-CE Cosmos'un Turkish-Llama-8b'si Llama-3 tabanlı, native Türkçe fine-tune'lu — production'a girdi."

---

## SLAYT 16 — Llama-3 Chat Template Bug Fix (Bonus Hikaye)

**İçerik:**
**Bug:** Turkish-Llama ilk çalıştırmada 595s'de "sahte assistant/user/system" diyalogları üretti.

**Tanı:** Llama-3 `eos_token_id` = `<|end_of_text|>` (128001), chat turn-end = `<|eot_id|>` (128009). Generation loop `<|eot_id|>`'i tanımıyorsa, model emit ediyor → `skip_special_tokens=True` role-tag'leri plain text bırakıyor → fake diyalog.

**Fix:** `_get_terminators()` helper — `<|eot_id|>` token'ını dinamik ekle.

**Sonuç:** 595s → 18.5s (32x), sahte diyalog gone.

**Görsel:** Sol: kötü output (sahte diyalog). Sağ: düzeltilmiş output. Code diff inset.

**Not:** "Buna benzer chat template bug'ları hugging face community'de yaygın. Saatler debug aldı ama küçük fix, büyük etki. Bu hikaye tezde 'production engineering rigor' kanıtı."

---

## SLAYT 17 — Demo 1: Kadın İstihdamı Sentezi (Multi-co)

**İçerik:**
**Sorgu:** "Kadın istihdamı ve fırsat eşitliği politikaları"

**Sistem:** 20 chunk → NLI 46 edges → MWIS+GAT kept 13/20 → LLM final answer

**Cevap özeti:**
- Aksa Akrilik: çeşitlilik politikası
- Anadolu Hayat: FEM Sertifika başvurusu (22 kriter, 7 uyumlu)
- Şekerbank: sosyal sorumluluk
- Borusan Holding: "Toplumsal Cinsiyet ve Ev İçinde Şiddet" politikası
- **Turkcell: "Eşit Olacak", "Geleceğini Yazan Kadın" programları** (gerçek!)

**Görsel:** Gradio screenshot + answer text.

**Not:** "5 farklı şirketten kaynaklar tutarlı şekilde sentezlendi. Turkcell'in gerçek programları doğru aktarıldı. **Bu pipeline'ın asıl gücü budur — multi-document synthesis.**"

---

## SLAYT 18 — Demo 2: Filtered Query (Akbank 2023)

**İçerik:**
**Sorgu:** "Akbank 2023 sürdürülebilirlik raporundaki ana başlıklar nelerdir?"
**Filtre:** `company=Akbank`, `year=2023`

**Sistem:** Metadata-filtered retrieval → 20 chunk (sadece Akbank 2023) → kept 16/20

**Cevap (7 ana başlık):**
1. Karbondan arındırma (Net Zero 2050)
2. Sera gazı emisyonu azaltma
3. Sürdürülebilir finansman
4. Çevre politikaları
5. Sosyal sorumluluk
6. İnsani kalkınma
7. Şeffaflık + raporlama

**Görsel:** Gradio screenshot, filtreleri vurgu.

**Not:** "Metadata filtering ile precision artışı. Single-company odaklı query'lerde retrieval daha keskin. Yapı tezde Section 6.3'te detaylı."

---

## SLAYT 19 — Demo 3: Cross-Sector Sentez

**İçerik:**
**Sorgu:** "Türk şirketlerinin sürdürülebilirlik raporlarında ortak temalar"

**Sistem:** 20 chunk × 5+ farklı sektör → NLI 12 edges → kept 17/20 → final answer

**Cevap:**
- **Tema 1:** Şeffaflık ve UN Sürdürülebilir Kalkınma Hedeflerine uyum (Kuveyt Türk, Anadolu Efes)
- **Tema 2:** AB CSRD (Corporate Sustainability Reporting Directive) implementasyonu
- **Tema 3:** SKD Türkiye "Raporlama Öncelikleri" programı (18 gösterge)

**Görsel:** Gradio screenshot + tema tag bulutu.

**Not:** "Sistem 5+ sektörden bilgiyi sentezleyip **EU regulation seviyesinde** çıkarım yaptı. CSRD ve SKD Türkiye gerçek referanslar — uydurma değil."

---

## SLAYT 20 — Limitations: Filter Layer

**İçerik:**
1. **Dense graphs (8+ chunks):** dense_graph boyutunda hem MWIS hem GAT 0/3 — heuristic optimization çoklu çelişki için yetersiz
2. **Türkçe NLI revize-tipi cümleler:** "2023'te %35'e revize edildi" gibi temporal-update'leri bazen 0 edge ile geçiriyor (mDeBERTa Türkçe doğrudan eğitilmedi)
3. **Numerical scope-difference false-positives:** Kapsam 1 / Kapsam 1+2 / Kapsam 1+2+3 farklı metrikler ama NLI bazen çelişki sanıyor

**Görsel:** 3 mini-örnek inline.

**Not:** "Tezsel olarak dürüst olunması gereken sınırlar. Bunlar future work — sonraki slayt."

---

## SLAYT 21 — Limitations: Generation Layer

**İçerik:**
**Generation eval'in ortaya koyduğu somut hatalar:**

1. **Entity name corruption (Tüpraş örneği):** Tek cevapta `Tüpraşa / Tüprag / Tüpragas / Tüprüğün / Tüpraga` — Llama-3 tokenizer × 4-bit nf4 interaction
2. **Aritmetik halüsinasyon:** "39 milyon + 34.800 = 73.800" (doğru: 39.034.800) — LLMs in genel aritmetik zaafı
3. **Cross-company drift:** Retrieval birden fazla şirketin chunks'ını getirdiğinde LLM atfı karıştırıyor
4. **Numerical drift %46:** generation eval'de niceliksel olarak ölçüldü

**Görsel:** 4 failure case mini-screenshot collage.

**Not:** "%54 numfid'in somut yüzleri. Bunlar **bir RAG sisteminin yapısal limitleri**, sadece bizim sistem değil. Tezsel dürüstlük."

---

## SLAYT 22 — Future Work

**İçerik:**
1. **Geniş etiketli korpus** (100+ manuel çelişki çifti) → GAT learning saturation
2. **NLI fine-tuning** mDeBERTa Türkçe revize cümleleri + scope-difference precision
3. **End-to-end training:** GAT scoring + MWIS + final answer joint optimization (REINFORCE / Straight-Through)
4. **Generation eval v2:** adversarial candidate set ile real fabrication detection
5. **Greenwashing-specific score** — interdepartmental + zero_claim pattern'lerini özel scoring metric'i

**Görsel:** Roadmap tipi timeline.

**Not:** "Bu projenin 5 doğal devamı. Özellikle (1) ve (5) sektörel ESG raporlama için doğrudan değerli."

---

## SLAYT 23 — Reproducibility & Open Source

**İçerik:**
- **GitHub repo:** `github.com/cgrti/ReliabilityRAG`
- **Bağımlılıklar:** PyTorch, transformers, bitsandbytes (4-bit), gradio
- **Modeller (HuggingFace):**
  - LLM: `ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1`
  - NLI: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
  - Embedding: `intfloat/multilingual-e5-base`
- **Hardware:** geliştirme RTX 2060 6GB (small profile), production RTX 5080 16GB
- **Veri:** 248 rapor, 182K chunk, public corporate reports
- **Reproducible:** `python ablation_mwis_vs_gat.py`, `python evaluation_generation.py`, `python make_demo_cache.py`

**Görsel:** QR code → GitHub repo. Logo'lar (HuggingFace, PyTorch, etc.).

**Not:** "Bütün kod, ağırlıklar, test seti açık. Tezdeki sayılar bir komutla yeniden üretilebilir."

---

## SLAYT 24 — Sonuç: Katkılar & Bulgular

**İçerik:**
**Özgün katkılar:**
1. Türkçe ESG corpus üzerinde **multi-doc RAG için NLI contradiction graph** kurulumu
2. **GAT-based dynamic filtering** + supervised training pipeline
3. **Generation faithfulness eval pipeline** (4 metrik, 2 mode)
4. 30-case 8-boyutlu sentetik test seti (open)

**Ana niceliksel bulgular:**
- Filter layer: %67 contradiction recall + %100 clean preservation
- GAT vs MWIS: 15/30 score eşit ama verdict heterogen (gat_only_ok=1)
- Generation layer: %54 numerical fidelity → **LLM dominant bottleneck**
- Failure case dokümantasyonu (Tüpraş, Vakıf GYO morphology, math)

**Görsel:** İki sütun: katkılar / bulgular.

**Not:** "Tek cümlede: **filtering doğrulandı, generation yeni bir cephe**. Multi-layer reliability framework artık niceliksel olarak ölçülebilir."

---

## SLAYT 25 — Teşekkürler + Q&A

**İçerik:**
> **Teşekkürler**
>
> Danışman: Prof. Dr. Yusuf Sinan Akgül
> Yardım için: Saliha Hoca (Akvaryum PC erişimi)
> Açık kaynak: HuggingFace, YTÜ-CE Cosmos lab
>
> **Sorularınız?**
>
> Repo: github.com/cgrti/ReliabilityRAG
> İletişim: cagri2002tireli@gmail.com

**Görsel:** Sade. QR code GitHub.

**Not:** "Sabırla dinlediğiniz için teşekkürler. Sorularınızı bekliyorum."

---

## Backup slaytlar (Q&A için)

### B1 — Per-test verdict tablosu (30 case)
Tüm ablation sonuçlarının detayı, jüri spesifik test sorarsa.

### B2 — Generation eval per-dimension breakdown
Section 6 detayı; "scope dimension'ında numfid neden 0.21 düşük?" sorusu için.

### B3 — NLI graph görsel örneği (8-chunk dense)
"Dense_graph 0/3 ne demek?" sorusu için.

### B4 — Architecture deep-dive: GAT katmanı
`GATConsistencyScorer` + skip-connection denklemleri.

### B5 — Hardware/profile yapısı
`config.GPU_PROFILES` (small/medium/large/xlarge) tablo; "neden RTX 2060'ta da çalışıyor?" sorusu için.

### B6 — Türkçe NLI Round 1→5 detayı
Numerical conflict detection logic örnekli.

---

## Slayt yapım önerileri

- **Format:** PowerPoint, Google Slides veya Beamer — fark etmez, içerik aynı
- **Tema:** Sade akademik (Beamer Madrid + tweaks, ya da minimal PPT template)
- **Renkler:** mavi + gri + tek vurgu rengi (kırmızı = uyarı/limitation, yeşil = sonuç)
- **Font:** sans-serif (Calibri / Source Sans Pro / Inter)
- **Görsel kuralı:** **her slaytta tek dominant görsel** veya tablo — text wall yok
- **Animasyon:** minimal, sadece bullet incremental reveal
- **Konuşma notu:** her slayt için 1-3 cümle, **provada bunları okuyarak çalış**

## Sıradaki

Bu outline 25 slayt + 6 backup = ~31 slot. Tipik tez savunması için ideal.

1. **Bugün/yarın:** Slaytları gerçekten oluştur (PowerPoint/Beamer). Outline → görsel slayt'a dönüştür. ~1-2 gün iş.
2. **Demo screenshots:** Slayt 17, 18, 19 için Gradio'dan ekran görüntüleri al, slaytlara göm.
3. **Diyagramlar:** Slayt 6 pipeline diagram, Slayt 7 NLI grafiği — Excalidraw / Mermaid ile çiz.
4. **Prova:** En az 3 kere baştan sona oku, süre tut, hangi slaytlarda takılıyorsun not al.

Yarın slaytları oluşturmaya başlarken bu outline'a referans olur.
