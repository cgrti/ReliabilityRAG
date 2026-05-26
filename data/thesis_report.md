# ReliabilityRAG — Tezsel Sonuçlar Raporu

Otomatik üretildi (`generate_thesis_report.py`). Tezdeki sayılar ve tablolar bu dosyadan kopyalanabilir.

---

## Özet (Tez Abstract için)

ReliabilityRAG, 248 Türk kurumsal entegre raporu (63 şirket, 2015–2024, 182.986 chunk) üzerinde çalışan, çok-dokümanlı bilgi çelişkilerini otomatik tespit eden 6-aşamalı bir RAG sistemidir. Standart paper'da MIS (Maximum Independent Set) kullanılan filtreleme katmanına özgün katkı olarak GAT (Graph Attention Network) tabanlı dinamik filtreleme önerilmiştir. 14-örnek sentetik test seti üzerinde yapılan ablation, iteratif kalibrasyon roundları sonrası MWIS baseline'ın **%67 filtreleme recall** ve **%100 temiz chunk korunması** sağladığını göstermiştir. Eğitilmiş GAT katmanı için BCE + contrastive margin loss tabanlı supervised training pipeline'ı (LOOCV, early stopping, weight decay) kurulmuş; üç hyperparametre rejimi test edilerek modelin *constant-collapse*, *plateau* ve *stable* davranışları gözlemlenmiştir. Eğitilmiş GAT mevcut datasette MWIS ile eşit performans (8/14) göstermiş, regresyon yapmamıştır. Sınırlama: dataset çeşitliliği GAT'ın learning capacity'sini doyurmamaktadır; gerçek dünya çelişki korpusu ile re-training future work olarak planlanmıştır.


## 1. Sentetik Test Seti

Çalışma boyunca kullanılan sentetik test seti **19 test case** içermektedir. Her test case bir soru, ilgili chunk seti ve ground-truth `expected_kept` / `expected_removed` etiketlerinden oluşur.

| Boyut | Test Sayısı | Açıklama |
|-------|-------------|----------|
| gat_discriminating | 5 | MWIS reliability-baskın yanılır, GAT recency öğrenmeli |
| temporal | 3 | Aynı şirketin yıllar arası farklı hedef/değer açıklamaları |
| scope | 3 | Aynı metriğin farklı kapsam/methodoloji ile farklı sayılar |
| interdepartmental | 3 | Strateji vs Finansal/Yönetim çelişkisi (greenwashing) |
| cross_company | 3 | — |
| zero_claim | 2 | — |

**Toplam:** 45 chunk · 19 çelişkili (ground truth) · 26 kept-target · 19 removed-target

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


## 6. Limitations & Future Work

### Veri sınırlamaları
- Sentetik test seti 14 case, çoğunluğu MWIS reliability-ranking ile çözülebilir
  yapıdadır; GAT'ın temporal-provenance + section-reliability feature'larını
  kullanarak ayrışacak nüans bu dataset'te yetersizdir.
- 3 test (`Test 1, 3, 5`) NLI'nin Türkçe revize-tipi cümleleri yakalayamaması
  nedeniyle 0 contradiction edge ile düşmektedir → ne MWIS ne GAT bu testlerde
  filtreleme yapamaz.

### Mimari sınırlamalar
- Pure GAT attention aggregation, bağlı node çiftlerinin feature'larını ortalar
  (skor collapse). Bu durum **skip connection** ile (blend=0.3 static + 0.7 GAT)
  hafifletildi.
- Heuristic-init zayıf bir lokal optimum yaratıyor; supervised training pipeline
  doğrulandı ancak küçük dataset bu optimumdan çıkışı sağlayamadı.

### Future work
1. **Geniş etiketli dataset**: 248 entegre rapordan manuel olarak 100+ çelişki
   pair'i etiketlemek. Beklenen kazanım: GAT learnable parameters için yeterli
   sinyal, MWIS-baseline'dan ayrışma.
2. **NLI fine-tuning**: mDeBERTa-v3-base'i Türkçe çelişki örnekleri üzerinde
   fine-tune etmek; özellikle "revize edildi" tipinde temporal-update cümleleri.
3. **End-to-end training**: GAT scoring + MWIS selection + final answer
   generation'ı birlikte optimize etmek (REINFORCE veya straight-through
   estimator ile).

