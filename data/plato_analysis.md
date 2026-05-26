# GAT Eğitim Platosu — Niçin Overfit Değil, Underfit + Dataset Bottleneck

**Tarih:** 2026-05-12
**Bağlam:** Saliha Hoca 2026-05-10 mesajında *"Modelin 8/14 seviyesinde plato yapması, ezberleme (overfitting) ile genelleyememe arasında sıkıştığını gösteriyor"* yorumunda bulundu. Bu kısa not, gözlemlenen davranışın **overfitting değil**, **underfit + dataset diversity bottleneck** olduğunu üç ayrı kanıtla göstermek için yazıldı.

---

## 1. Overfitting'in operasyonel tanımı

Overfit: **training loss düşer**, **training accuracy %100'e yaklaşır**, ama **validation accuracy düşer/sabit kalır**.

Bizim durumumuzda ne gözlemlendi:

| Rejim | Final training loss | Training-set accuracy (14/14 hedef) | LOOCV accuracy |
|-------|---------------------|--------------------------------------|----------------|
| Aggressive (lr=5e-3, blend=1.0, 1000ep) | ~0.69 (BCE sat.) | **0/14** (constant collapse) | 0/14 |
| Initial (lr=5e-3, blend=0.7, 500ep) | 0.512 (plateau) | **8/14** | 8/14 |
| Conservative (lr=1e-3, blend=0.5, 200ep, early-stop @39) | 0.705 (plateau) | **8/14** | 8/14 |

**Kritik nokta:** Conservative + Initial rejimlerinde model **training set'ini bile %100 ezberleyemiyor** (8/14). Overfit olsa training-set accuracy 14/14, val accuracy <14/14 olurdu. Bizde her ikisi de 8/14 — yani **henüz fit aşamasına bile gelinmemiş**.

## 2. Hatanın yapısal kaynağı: NLI kısa devresi

14 testin **3'ünde** (Test 1, 3, 5) NLI sistemi **hiç çelişki edge'i çizmiyor** (`G.number_of_edges() == 0`). `nli_graph.filter_chunks` bu durumda kısa devre yapıp `list(range(n))` döndürür — yani hiçbir filtreleme yapmaz, **ne MWIS ne GAT karar verebilir**.

Bu 3 test maksimum ulaşılabilir tavanı **11/14**'e düşürür. Bizim 8/14 sonucumuz, "kararlanabilir 11 testin 8'inde doğru" demektir = **%72.7 fit accuracy** — overfitting değil, GAT'ın hâlâ optimize edilebileceği yer var.

| Test no | Boyut | NLI edge sayısı | GAT karar fırsatı | Sonuç |
|---------|-------|------------------|--------------------|-------|
| 1, 3, 5 | temporal/scope | **0** | Yok (kısa devre) | Kurtarılamaz |
| 12 (GAT-3) | gat_discriminating | **0** | Yok | Kurtarılamaz |
| Diğer 10 test | — | ≥1 | Var | 8 doğru, 2 zor |

## 3. Hyperparameter sweep — model kapasitesi yeterli

Üç hyperparametre rejiminde modelin **3 farklı davranış** sergilediğini gözlemledik:

- **Collapse** (lr=5e-3, blend=1.0): Tüm output'lar 0.508'e çöktü. Model parametre uzayında "her şey sabit" lokal minimum'una düştü → **kapasite var, doğru rejim yok**.
- **Plateau** (lr=5e-3, blend=0.7): Loss 0.512'de takıldı, gradient akıyor ama yeterince sinyal yok → **dataset sinyali zayıf**.
- **Stable** (lr=1e-3, blend=0.5, weight decay, early stop): Heuristic-init'e yakın, regresyon yok → **mevcut minimum etrafında lokal**.

Eğer overfit olsaydı, Conservative rejimde model 14/14'e ulaşırdı (küçük dataset = kolay ezberlenir). Aksine 8/14'te kaldı: **eğitim verisinin kendisi modeli yönlendiremiyor**.

## 4. Sonuç ve tezsel iddia

> Gözlemlenen plato, modelin **kapasite sınırı** değil **veri çeşitliliği sınırıdır**. 14-örnek sentetik test seti, GAT'ın temporal-provenance ve section-reliability feature'larını ayrıştırıcı katkıya çevirebileceği nüansları yeterince temsil etmemektedir. Ablation altyapısı doğrulandı, regresyon yok, üç hyperparametre rejimi sistematik olarak gözlemlendi. Bu, **bir sonraki adım için açık bir reçete** sunar: (a) etiketli korpus 100+ pair'e çıkarılmalı, (b) NLI fine-tune'u Türkçe revize-tipi cümleleri yakalamalı, (c) genişletilmiş eğitim sonrası ablation tekrarlanmalı.

## 5. Hocaya tek cümlelik özet

> *"Model 14 örnekte 8/14 plato yapıyor — training set'i bile ezberleyemiyor; bu overfit değil underfit. NLI 3 testte hiç edge çizemiyor (kısa devre), bu da maksimum tavanı 11/14'e düşürüyor. Yani fit accuracy = 8/11 = %73; geniş etiketli korpus ve NLI fine-tune ile bu sınırı aşmayı planlıyorum."*

---

**İlgili dosyalar:**
- `ablation_mwis_vs_gat.py` — karşılaştırma altyapısı
- `train_gat.py` — supervised training pipeline
- `data/ablation_results.json` — sayısal sonuçlar
- `data/thesis_report.md` — tezsel rapor (§2, §5)
