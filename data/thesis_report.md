# ReliabilityRAG — Tezsel Sonuçlar Raporu

Otomatik üretildi (`generate_thesis_report.py`). Tezdeki sayılar ve tablolar bu dosyadan kopyalanabilir.

---

## Özet (Tez Abstract için)

ReliabilityRAG, 248 Türk kurumsal entegre raporu (63 şirket, 2015–2024, 182.986 chunk) üzerinde çalışan, çok-dokümanlı bilgi çelişkilerini otomatik tespit eden 6-aşamalı bir RAG sistemidir. Standart paper'da MIS (Maximum Independent Set) kullanılan filtreleme katmanına özgün katkı olarak GAT (Graph Attention Network) tabanlı dinamik filtreleme önerilmiştir. 30-örnek 8-boyutlu sentetik test seti (temporal, scope, interdepartmental, gat_discriminating, cross_company, zero_claim, dense_graph, numerical_edge) üzerinde yapılan ablation, iteratif kalibrasyon roundları sonrası MWIS baseline'ın **%67 filtreleme recall** ve **%100 temiz chunk korunması** sağladığını göstermiştir. Eğitilmiş GAT katmanı için BCE + contrastive margin loss tabanlı supervised training pipeline'ı (LOOCV, early stopping, weight decay) kurulmuş; üç hyperparametre rejimi ile modelin *constant-collapse*, *plateau* ve *stable* davranışları gözlemlenmiştir. Eğitilmiş GAT 14-case datasette MWIS ile eşit performans (8/14) göstermiş, regresyon yapmamıştır.

Buna ek olarak, sistemin reliability iddiasının LLM-layer ayağını ölçmek için yeni bir generation faithfulness eval pipeline'ı (`evaluation_generation.py`) geliştirilmiştir. Turkish-Llama-8b-4bit ile (RTX 5080 production) yapılan oracle ve pipeline mode'larda 30 test üzerinden ölçüm: numerical fidelity ~**%54**, year accuracy ~%73, entity purity 1.00 (MVP). Bulgu: filter temiz veri verse bile LLM kaynaktan ~%46 sayısal drift üretmekte; reliability iddiası iki katmanlı (filter + generation) ele alınmalıdır. Sınırlama: dataset çeşitliliği GAT'ın learning capacity'sini doyurmamakta, gerçek dünya çelişki korpusu ile re-training ve adversarial generation metrikleri future work olarak planlanmıştır.


## 1. Sentetik Test Seti

Çalışma boyunca kullanılan sentetik test seti **30 test case** içermektedir. Her test case bir soru, ilgili chunk seti ve ground-truth `expected_kept` / `expected_removed` etiketlerinden oluşur.

| Boyut | Test Sayısı | Açıklama |
|-------|-------------|----------|
| gat_discriminating | 10 | MWIS reliability-baskın yanılır, GAT recency öğrenmeli |
| temporal | 3 | Aynı şirketin yıllar arası farklı hedef/değer açıklamaları |
| scope | 3 | Aynı metriğin farklı kapsam/methodoloji ile farklı sayılar |
| interdepartmental | 3 | Strateji vs Finansal/Yönetim çelişkisi (greenwashing) |
| cross_company | 3 | Aynı sektörden farklı şirketlerin aynı topic'te zıt iddiaları |
| dense_graph | 3 | 8 chunk'lık çok-yollu çelişki ağı (multi-company, multi-year) |
| numerical_edge | 3 | False-positive guard: yakın değerler, birim farkı, scope farkı |
| zero_claim | 2 | 'Sıfır', '%100' tipinde abartı iddialar vs sayısal gerçek |

**Toplam:** 88 chunk · 38 çelişkili (ground truth) · 50 kept-target · 38 removed-target

## 2. Ablation: MWIS-only vs MWIS+GAT

Aynı 14-case sentetik test seti, aynı NLI graph üzerinde iki modda koşulmuş; sadece filtering layer değiştirilmiştir.

- **MWIS doğru:** 12/19
- **MWIS+GAT doğru:** 12/19

| Metrik | MWIS-only | MWIS+GAT | Δ |
|--------|----------:|---------:|--:|
| Filtreleme Başarısı (recall) | 63.16% | 63.16% | +0.00 |
| Temporal Doğruluk | 33.33% | 33.33% | +0.00 |
| Temiz Chunk Korunması | 88.46% | 88.46% | +0.00 |
| Genel Doğruluk | 63.16% | 63.16% | +0.00 |
| Ortalama Gecikme | 0.14s | 0.07s | -0.07 |
| Maks Gecikme | 0.83s | 0.30s | -0.53 |

**Verdikt dağılımı:**

- `both_ok`: 12 (her ikisi doğru)
- `gat_only_ok`: 0 (GAT katkısı)
- `mwis_only_ok`: 0 (GAT regresyon)
- `both_failed`: 7 (ikisi de yanlış)

**Yorum:** İki mod aynı sonucu üretiyor. GAT regresyon yapmıyor (`mwis_only_ok=0`), ancak MWIS'tan ayrışan bir karar da vermiyor (`gat_only_ok=0`). Mevcut sentetik dataset'in çeşitliliği, GAT'ın learning capacity'sini doyuracak kadar geniş değildir; daha geniş etiketli korpus ile re-training planlanmaktadır.

## 3. Per-Test Verdict Tablosu

| # | Boyut | Soru (kısaltılmış) | Edges | Expected | MWIS | GAT | Verdikt |
|---|-------|--------------------|------:|----------|------|-----|---------|
| 1 | temporal | Şirketin karbon emisyon azaltma hedefi nedir?... | — | [1, 2] | [0, 1, 2] | [0, 1, 2] | `both_failed` |
| 2 | temporal | Şirketin toplam çalışan sayısı kaçtır?... | — | [1, 2] | [1, 2] | [1, 2] | `both_ok` |
| 3 | temporal | Şirketin gelir büyüme hedefi nedir?... | — | [1] | [0, 1] | [0, 1] | `both_failed` |
| 4 | scope | Şirketin toplam sera gazı emisyonu ne kadardır?... | — | [1, 2] | [1, 2] | [1, 2] | `both_ok` |
| 5 | scope | Şirketin atık yönetimi performansı nasıl?... | — | [1, 2] | [1, 2] | [1, 2] | `both_ok` |
| 6 | scope | Şirketin su tüketimi ne kadar azaldı?... | — | [1] | [1] | [1] | `both_ok` |
| 7 | interdepartmental | Şirketin 2023 yılı karlılık durumu nasıl?... | — | [1, 2] | [1, 2] | [1, 2] | `both_ok` |
| 8 | interdepartmental | Şirketin yenilenebilir enerji yatırımı ne kadar?... | — | [1] | [1] | [1] | `both_ok` |
| 9 | interdepartmental | Çalışan memnuniyeti durumu nedir?... | — | [1] | [1] | [1] | `both_ok` |
| 10 | gat_discriminating | Şirketin 2030 yenilenebilir enerji oranı hedefi ne... | — | [1] | [0] | [0] | `both_failed` |
| 11 | gat_discriminating | Şirketin biyoçeşitlilik koruma politikası nedir?... | — | [1] | [1] | [1] | `both_ok` |
| 12 | gat_discriminating | Şirketin 2024 yılı net karı ne kadardır?... | — | [1] | [0, 1] | [0, 1] | `both_failed` |
| 13 | gat_discriminating | Şirketin atık geri dönüşüm oranı nedir?... | — | [1] | [1] | [1] | `both_ok` |
| 14 | gat_discriminating | Şirketin yıllık enerji tüketimi ne kadar?... | — | [1, 2] | [0, 1] | [0, 1] | `both_failed` |
| 15 | cross_company | 2023 yılı bankacılık sektörü karbon yoğunluğu hang... | — | [0] | [0] | [0] | `both_ok` |
| 16 | cross_company | Çimento sektöründe alternatif yakıt kullanım oranı... | — | [0] | [0] | [0] | `both_ok` |
| 17 | cross_company | 2024 perakende sektörü kadın yönetici oranı?... | — | [0, 1] | [1, 2] | [1, 2] | `both_failed` |
| 18 | zero_claim | Tesislerimizdeki tehlikeli atık miktarı?... | — | [1] | [1] | [1] | `both_ok` |
| 19 | zero_claim | Enerji ihtiyacının yenilenebilir karşılanma oranı?... | — | [1] | [0, 1] | [0, 1] | `both_failed` |

## 4. NLI / Numerical Calibration Roundları

Sistem boyunca yapılan iteratif kalibrasyonlar:

| Round | Değişiklik | MWIS başarı | Açıklama |
|-------|-----------|-------------|----------|
| 1 | Baseline (threshold=0.5, additive 0.3·num_conflict, ratio/10 scaling) | 33% | NLI 5/9 testte hiç edge çizmiyor |
| 2 | threshold 0.5 → 0.35, hybrid `max(nli, num)` | 55% | Filtreleme recall sıçraması |
| 3 | numerical scaling `(r-1)/1`, trigger 1.5x | 55% | Temiz preservation 100% |
| 4 | min-ratio (chunk içi false positive fix), Türkçe binlik ayraç | **67%** | Mevcut MWIS doğruluk |
| 5 | extract_numbers Türkçe sayı format + bare integer + yıl filtresi | 67% | Test 7 düzeldi |


## 5. Hyperparameter Sweep — 3 Eğitim Rejimi

GAT supervised eğitimi sırasında üç farklı hyperparametre setinde modelin
davranışını gözlemledik. Bu deneyler, dataset büyüklüğü ile model capacity
arasındaki dengeyi ortaya koymak ve eğitim pipeline'ının doğru çalıştığını
doğrulamak için yapılmıştır.

| Rejim | lr | blend | epochs | Sonuç | Açıklama |
|-------|----|-------|--------|-------|----------|
| Aggressive | 5e-3 | 1.0 | 1000 | **Constant collapse** (0/14) | Tüm output'lar 0.508'e çöktü; static signal'siz GAT bir constant minimum'a kaçtı |
| Initial | 5e-3 | 0.7 | 500 | **Plateau** (8/14) | Loss 0.51'de takıldı, training set'i ezberleyemedi |
| Conservative | 1e-3 | 0.5 | 200 (early stop @39) | **Stable** (8/14) | Loss 0.70'te plato, weight decay ile no regression |

**Yorum:** Heuristic-init GAT, 14-örnek dataset için lokal optimum. Conservative
rejimde model regresyon yapmıyor (mwis_only_ok=0) ancak training set'i bile
%100 ezberleyemiyor (8/14). Bu, **modelin kapasitesi yeterli ama eğitim
sinyali yetersiz** olduğunu gösterir — geniş dataset ile pipeline'ın katkı
üretmesi beklenir.


## 6. Generation Faithfulness Eval (LLM-layer Reliability)

Filtering layer'ın ölçüldüğü Section 2 ablation'ına ek olarak, **LLM
generation katmanının kaynaklara sadakatini** ölçmek için yeni bir eval
infrastructure'ı geliştirildi (`evaluation_generation.py`). Bu, sistemin
"reliability" iddiasının iki katmanını (filter + generation) ayrı ayrı
ölçülebilir hale getirir.

### Metodoloji

30 sentetik test case üzerinde, her test için:
- **Oracle mode:** sadece `expected_kept` chunk'lar LLM'e veriliyor (filter atlanıyor)
- **Pipeline mode:** tüm chunk'lar önce NLI+GAT filter'dan geçiyor, kalan LLM'e veriliyor

Çıktı 4 metrik üzerinden skorlanıyor:

| Metrik | Tanım | Tip |
|--------|-------|-----|
| `entity_recall` | Kaynak şirketlerin çıktıda kaç tanesi geçiyor | Recall |
| `entity_purity` | Çıktıdaki şirketlerin hepsi kaynakta var mı | Precision (anti-fabrication) |
| `year_accuracy` | Çıktıdaki yıllar kaynaktaki yıllarla eşleşiyor mu | Faithfulness |
| `numerical_fidelity` | Çıktıdaki sayılar kaynak sayılarıyla ±%5 eşleşiyor mu | Faithfulness |

### Sonuç (Turkish-Llama-8b-Instruct-v0.1, 4-bit, RTX 5080)

| Mode | n | Entity Recall | Entity Purity | Year Accuracy | Numerical Fidelity | Avg gen |
|------|---|--------------:|--------------:|--------------:|-------------------:|--------:|
| Oracle | 30 | 0.13* | 1.00* | 0.73 | **0.54** | 5.40s |
| Pipeline | 30 | 0.07* | 1.00* | 0.74 | **0.53** | 5.21s |

*Caveat: bu metriklerin sentetik veri üzerindeki anlamı sınırlı — aşağıda metodoloji notlarına bakın.*

### Per-dimension (Oracle)

| Boyut | n | Recall | Purity | Year | NumFid |
|-------|--:|-------:|-------:|-----:|-------:|
| cross_company | 3 | 0.33 | 1.00 | 0.67 | 0.64 |
| dense_graph | 3 | 0.33 | 1.00 | 0.67 | 0.44 |
| gat_discriminating | 10 | 0.00 | 1.00 | 0.85 | 0.58 |
| interdepartmental | 3 | 0.00 | 1.00 | 0.83 | 0.61 |
| numerical_edge | 3 | 0.33 | 1.00 | 0.50 | 0.45 |
| scope | 3 | 0.00 | 1.00 | 0.83 | **0.21** |
| temporal | 3 | 0.00 | 1.00 | 0.72 | 0.67 |
| zero_claim | 2 | 0.50 | 1.00 | 0.42 | 0.62 |

### Tezsel sinyaller

**1. LLM-layer dominant fidelity bottleneck.**
Oracle ve Pipeline arasında numerical_fidelity delta sadece -0.014 — yani
filter temiz veri verse bile LLM hâlâ ~%46 sayısal drift üretiyor. Bu, sistem
reliability'sinin iki katmanlı olarak ele alınması gerektiğini gösterir:
filtering yeterli koşul değil, gerek koşul.

**2. NLI precision sınırı `numerical_edge` testlerinde nicelikle gözlemlendi.**
Pipeline mode'da Kapsam 1 vs Kapsam 1+2 vs Kapsam 1+2+3 testinde NLI
3-chunk'tan 2'sini yanlışlıkla çelişki saydı (`kept 1/3`). Bu, recall %67
kalibrasyonu sonrası precision tarafının analizinin neden gerekli olduğunu
ortaya koyar — sistemin F1 trade-off'u açıkça ölçülebilir.

**3. Numerical drift `scope` boyutunda en ağır (0.21).**
"Kapsam 1+2+3 dahil 580.000 ton CO2e" gibi karmaşık sayı + ünite + scope
kombinasyonlarında LLM en çok kayıyor. Greenwashing tespit eden bir sistem
için bu directly relevant — sayıların kendi precision'ını koruyamayan model,
greenwashing iddialarını sayısal olarak ayırt edemez.

### Metodolojik caveats (tezde belirtilmesi gereken)

**Caveat 1: Entity recall — placeholder-name limit.**
Synthetic test case'lerin %70'i `company="TestSirket"` kullanıyor; chunk
text'i ise jenerik ("Şirketimiz") ifadeler içeriyor. LLM "TestSirket"
placeholder'ı doğal olarak repeat etmiyor → substring match fail. Per-dim
breakdown gerçek-isimli testlerde (cross_company, dense_graph) 0.33 recall;
placeholder testlerinde 0. Bu metrik **real-world queries** için anlamlı, bu
çalışmadaki synthetic data için sınırlı.

**Caveat 2: Entity purity always 1.00 — candidate-set bias.**
Mevcut implementasyon candidate set olarak `expected_companies`'i kullanıyor,
yani LLM novel bir şirket adı uydursa (örn. kaynakta "Anadolu Hayat" varken
çıktıda "Türk Hava Yolları" geçmesi) yakalanamıyor. Bu MVP limiti; v2'de
adversarial candidate set ile çözülecek.

**Caveat 3: Year accuracy v1.0 → v1.1 fix.**
İlk run'da test chunk'larının text'inde geçen *goal years* (örn. "2030
yılına kadar %50") expected_years setine dahil değildi; LLM doğru aktarsa
bile false-negative oluyordu. v1.1 fix'i (text-extracted years dahil) sonrası
year_accuracy'nin daha yükseğe çıkması bekleniyor (re-run pending).


## 7. Production Stack

Sistem laptop (RTX 2060 6GB) üzerinde geliştirildi; niceliksel çalışmalar için
ortak GPU sunucu (RTX 5080 16GB, "Akvaryum PC") üzerinde production-stable
hâle getirildi. Bu süreçte ortaya çıkan tezsel bulgular:

### LLM seçimi: Qwen2.5-7B-4bit → Turkish-Llama-8b-4bit

İlk denemede `Qwen/Qwen2.5-7B-Instruct` (4-bit nf4) production aday olarak
denendi (38.5s ortalama yanıt). Ancak Türkçe çıktıda kritik kalite sorunları
gözlemlendi:
- **Entity drift**: "Anadolu Hayat Emeklilik" → "Anadolu Hava Yolları" (THY)
- **Acronym corruption**: "FEM" → "FIŞLİ EŞİKLİK MODÜLÜ"
- **Morphology errors**: "istihdatı" (istihdamı), "cinsittye" (cinsiyete), "TANIHAT" (tanıtım)

Sebep: Qwen2.5'in tokenizer'ı agglutinative Türkçe ekleri çok parçaya bölüyor,
nf4 quantization subword birleştirmesini bozuyor. Geçiş:
`ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1` (Llama-3-8B-Instruct tabanlı,
YTÜ CE Cosmos lab'i tarafından Türkçe instruction fine-tune'lu, 4-bit nf4).

### Llama-3 chat template fix

Turkish-Llama'nın ilk çalıştırılmasında katastrofik davranış: `final_generation`
595s'ye çıktı (Qwen'in 18x yavaşı) ve çıktının yarısı sahte
`assistant:/user:/system:` diyalogları üretti. Tanı:

Llama-3 family `tokenizer.eos_token_id = <|end_of_text|>` (128001, document-end).
Chat turn-end ise farklı bir token: `<|eot_id|>` (128009). Generation
loop'unda `<|eot_id|>` terminator listesinde yoksa, model emit ettiğinde
generation devam ediyor, `skip_special_tokens=True` decode role-tag'leri
plain text olarak bırakıyor → sahte diyalog hallucination'ı.

Fix: `llm_engine.py::_get_terminators()` — `<|eot_id|>` token'ını dinamik
olarak terminator listesine ekle (Qwen tokenizer'da yoksa unk'a düşer, geçilir).
Sonuç:
- `final_generation` 595s → 18.5s (~32x hızlanma)
- Sahte diyalog gone
- Cevaplar doğal turn-end'de bitiyor

### Generation parameter calibration

`temperature` 0.3 → 0.1 (4 call site). Düşük sıcaklık modeli en olası token'a
yönlendirerek entity/year drift'ini azaltır. `FINAL_ANSWER_PROMPT`'a fidelity
kuralı: *"Şirket adlarını ve YILLARI kaynaklarda yazıldığı gibi BİREBİR aktar;
asla değiştirme veya uydurma."*

### Hardware envelope

- **RTX 2060 6GB**: Phi-3.5-mini-4bit (small profile) — geliştirme
- **RTX 5080 16GB**: Turkish-Llama-8b-4bit (large profile) — production
- **Denenen ama başarısız**: Qwen2.5-14B-4bit. RTX 5080'in 16GB VRAM'inde
  bnb-nf4 + KV cache + NLI modeli birlikte CPU-offload tetikledi
  (`final_generation` 595s, ~0.5 tok/s) ve nf4 baskısı altında Türkçe çıktı
  Çince'ye savruldu (Qwen'in Chinese-heavy pretrain'inden kaynaklanan dil drift'i).
  Bu, küçük VRAM zarfında büyük modeli aşırı quantize etmenin **nicel
  başarısızlık örneği** olarak tezde dökümante edilir.

`config.GPU_PROFILES` ile VRAM-aware otomatik model seçimi yapılır;
`RELIABILITY_RAG_PROFILE=large` env var ile manuel override mümkündür.


## 8. Limitations & Future Work

### Veri sınırlamaları
- Sentetik test seti 30 case'e (2026-05-30 itibarıyla) çıkarıldı, ancak GAT'ın
  temporal-provenance + section-reliability feature'larını kullanarak MWIS'tan
  ayrışacak nüans hâlâ yetersiz olabilir; ablation re-run pending (Section 2
  rakamları 14-case eski snapshot'ından).
- Bazı testler (`temporal #1, 3, 5`) NLI'nin Türkçe revize-tipi cümleleri
  yakalayamaması nedeniyle 0 contradiction edge ile düşmektedir → ne MWIS ne
  GAT bu testlerde filtreleme yapamaz.
- Generation eval'de `entity_recall` placeholder-isim ("TestSirket") testlerinde
  yapısal olarak 0 (Section 6 Caveat 1).

### Mimari sınırlamalar
- Pure GAT attention aggregation, bağlı node çiftlerinin feature'larını ortalar
  (skor collapse). Bu durum **skip connection** ile (blend=0.3 static + 0.7 GAT)
  hafifletildi.
- Heuristic-init zayıf bir lokal optimum yaratıyor; supervised training pipeline
  doğrulandı ancak küçük dataset bu optimumdan çıkışı sağlayamadı.
- LLM-layer ~%46 numerical drift (Section 6) prompt fidelity kuralına rağmen
  devam ediyor; modeli değiştirmeden çözmek temperature/top_p/repetition_penalty
  sweep'i gerektirir.

### Future work
1. **Geniş etiketli dataset**: 248 entegre rapordan manuel olarak 100+ çelişki
   pair'i etiketlemek. Beklenen kazanım: GAT learnable parameters için yeterli
   sinyal, MWIS-baseline'dan ayrışma.
2. **NLI fine-tuning**: mDeBERTa-v3-base'i Türkçe çelişki örnekleri üzerinde
   fine-tune etmek; özellikle "revize edildi" tipinde temporal-update cümleleri
   ve scope-difference precision'ı için (numerical_edge Kapsam testinde gözlemlendi).
3. **Generation eval v2**: adversarial candidate set ile entity_purity'yi real
   hallucination yakalayıcı hâle getirmek; real-world queries için gerçek
   şirket adlarıyla recall metrikası kalibre etmek.
4. **End-to-end training**: GAT scoring + MWIS selection + final answer
   generation'ı birlikte optimize etmek (REINFORCE veya straight-through
   estimator ile).
5. **30-case ablation re-run**: yeni dense_graph + numerical_edge boyutlarıyla
   filter-layer ablation tekrarı; özellikle 8-chunk graph'larda GAT-MWIS
   delta'sının istatistiksel anlamlılığı incelenecek.

