"""
3-Dimensional Synthetic Inconsistency Test Set Generator

Generates test scenarios across 3 dimensions:
1. Temporal Target Deviations: Same target changes across years
2. Data Coverage Inconsistencies: Scope/metric mismatches
3. Interdepartmental Conflicts: Marketing vs Financial contradictions

Each test case has:
- A question
- A set of chunks (some consistent, some contradictory)
- Ground truth: which chunks should be KEPT and which REMOVED
- Expected behavior: what the system should detect

This enables measuring:
- Inconsistency Filtering Success (target: ≥90%)
- Temporal Consistency Accuracy (target: 100%)
- Source Fidelity (target: ≥95%)
"""
import json
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

from config import PROJECT_ROOT

TESTSET_PATH = PROJECT_ROOT / "data" / "synthetic_testset.json"


@dataclass
class TestChunk:
    text: str
    company: str
    year: int
    section_type: str
    reliability_weight: float
    is_contradictory: bool  # Ground truth: should this be filtered out?
    contradiction_type: str  # "temporal", "scope", "interdepartmental", "none"


@dataclass
class TestCase:
    test_id: str
    dimension: str  # "temporal", "scope", "interdepartmental"
    question: str
    chunks: list[TestChunk]
    expected_kept: list[int]  # Indices of chunks that SHOULD be kept
    expected_removed: list[int]  # Indices that SHOULD be removed
    description: str


def generate_temporal_tests() -> list[TestCase]:
    """
    Dimension 1: Temporal Target Deviations
    Same company changes targets/claims across years.
    System should prioritize the most recent report.
    """
    tests = []

    # Test 1: Carbon emission target changed
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="temporal",
        question="Şirketin karbon emisyon azaltma hedefi nedir?",
        chunks=[
            TestChunk(
                text="Şirketimiz 2030 yılına kadar karbon emisyonlarını %50 azaltma hedefi koymuştur. Bu hedef, Paris İklim Anlaşması doğrultusunda belirlenmiştir.",
                company="TestSirket", year=2021, section_type="cevre",
                reliability_weight=0.6, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="Stratejik planımız güncellenerek, 2030 karbon azaltım hedefimiz %50'den %35'e revize edilmiştir. Ekonomik koşullar ve teknolojik kısıtlar bu revizyonu zorunlu kılmıştır.",
                company="TestSirket", year=2023, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="2024 sürdürülebilirlik raporumuza göre, güncel karbon azaltım hedefimiz 2030 itibarıyla %35 olarak belirlenmiştir. Bu hedef doğrultusunda yenilenebilir enerji yatırımlarımız devam etmektedir.",
                company="TestSirket", year=2024, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1, 2],
        expected_removed=[0],
        description="2021'de %50 hedefi → 2023'te %35'e revize. Sistem en güncel hedefi (2023-2024: %35) tercih etmeli.",
    ))

    # Test 2: Employee count contradiction
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="temporal",
        question="Şirketin toplam çalışan sayısı kaçtır?",
        chunks=[
            TestChunk(
                text="Şirketimiz bünyesinde 2022 yıl sonu itibarıyla toplam 12.500 çalışan bulunmaktadır.",
                company="TestSirket", year=2022, section_type="sosyal",
                reliability_weight=0.5, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2023 yılında gerçekleştirdiğimiz organizasyonel dönüşüm kapsamında çalışan sayımız 8.200'e düşmüştür.",
                company="TestSirket", year=2023, section_type="sosyal",
                reliability_weight=0.5, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="İnsan kaynakları politikamız gereği, 2024 itibarıyla toplam 8.450 çalışanımız bulunmaktadır.",
                company="TestSirket", year=2024, section_type="sosyal",
                reliability_weight=0.5, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1, 2],
        expected_removed=[0],
        description="Çalışan sayısı 12500→8200→8450 değişimi. Eski veri (2022) elenmeli.",
    ))

    # Test 3: Revenue target shift
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="temporal",
        question="Şirketin gelir büyüme hedefi nedir?",
        chunks=[
            TestChunk(
                text="2025 stratejik planımızda yıllık %25 gelir büyümesi hedeflenmektedir. Bu hedef, pazardaki büyüme trendleri ve yatırım planlarımız doğrultusunda belirlenmiştir.",
                company="TestSirket", year=2020, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="Küresel ekonomik yavaşlama nedeniyle 2025 gelir büyüme hedefimiz %25'ten %12'ye revize edilmiştir.",
                company="TestSirket", year=2023, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1],
        expected_removed=[0],
        description="Gelir hedefi %25→%12 revize. Eski strateji (2020) elenmeli, güncel finansal (2023) kalmalı.",
    ))

    return tests


def generate_scope_tests() -> list[TestCase]:
    """
    Dimension 2: Data Coverage / Scope Inconsistencies
    Same metric reported with different scopes or methodologies.
    """
    tests = []

    # Test 1: Emission scope mismatch
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="scope",
        question="Şirketin toplam sera gazı emisyonu ne kadardır?",
        chunks=[
            TestChunk(
                text="Toplam sera gazı emisyonumuz 150.000 ton CO2e olarak gerçekleşmiştir. Bu rakam yalnızca Kapsam 1 (doğrudan) emisyonları içermektedir.",
                company="TestSirket", year=2023, section_type="cevre",
                reliability_weight=0.6, is_contradictory=True,
                contradiction_type="scope",
            ),
            TestChunk(
                text="2023 yılında toplam sera gazı emisyonumuz Kapsam 1, 2 ve 3 dahil 580.000 ton CO2e olarak hesaplanmıştır. Bağımsız doğrulama sürecinden geçmiştir.",
                company="TestSirket", year=2023, section_type="cevre",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="Çevresel performans tablomuzda gösterildiği üzere, doğrulanmış toplam emisyonumuz 580.000 ton CO2 eşdeğeridir.",
                company="TestSirket", year=2023, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1, 2],
        expected_removed=[0],
        description="150K (sadece Kapsam 1) vs 580K (Kapsam 1+2+3). Dar kapsamlı veri elenmeli.",
    ))

    # Test 2: Greenwashing — marketing vs financial reality
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="scope",
        question="Şirketin atık yönetimi performansı nasıl?",
        chunks=[
            TestChunk(
                text="Sıfır atık politikamız kapsamında tüm tesislerimizde atık minimizasyonu sağlanmış olup, üretim süreçlerimizde çevreye zararlı hiçbir atık oluşmamaktadır.",
                company="TestSirket", year=2023, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="scope",
            ),
            TestChunk(
                text="2023 yılı çevre denetim raporuna göre tesislerimizden toplam 45.000 ton tehlikeli atık bertaraf edilmiştir. Atık bertaraf giderleri bir önceki yıla göre %40 artmıştır.",
                company="TestSirket", year=2023, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="Çevre Bakanlığı denetimleri kapsamında tesislerimizde ölçülen atık miktarları yasal sınırlar dahilinde kalmaktadır. Tehlikeli atık miktarı 45.200 ton olarak raporlanmıştır.",
                company="TestSirket", year=2023, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1, 2],
        expected_removed=[0],
        description="Greenwashing: 'Sıfır atık' iddiası vs 45K ton tehlikeli atık gerçeği. Pazarlama söylemi elenmeli.",
    ))

    # Test 3: Water consumption different bases
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="scope",
        question="Şirketin su tüketimi ne kadar azaldı?",
        chunks=[
            TestChunk(
                text="Su tüketimimiz %60 azalmıştır. Yağmur suyu toplama sistemlerimiz büyük katkı sağlamıştır.",
                company="TestSirket", year=2023, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="scope",
            ),
            TestChunk(
                text="Toplam su çekimimiz 2022'de 2.1 milyon m³ iken 2023'te 1.85 milyon m³'e düşmüştür (%12 azalma). Azalma büyük ölçüde üretim hattı optimizasyonundan kaynaklanmaktadır.",
                company="TestSirket", year=2023, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1],
        expected_removed=[0],
        description="%60 azalma (tek tesis) vs %12 azalma (tüm operasyonlar). Abartılı iddia elenmeli.",
    ))

    return tests


def generate_interdepartmental_tests() -> list[TestCase]:
    """
    Dimension 3: Interdepartmental Conflicts
    Marketing/PR says one thing, financial statements say another.
    """
    tests = []

    # Test 1: Profitability claim vs actual loss
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="interdepartmental",
        question="Şirketin 2023 yılı karlılık durumu nasıl?",
        chunks=[
            TestChunk(
                text="2023 yılında güçlü operasyonel performansımız devam etmiş, tüm iş kollarımızda karlılığımızı artırmayı başardık. Sürdürülebilir büyüme stratejimiz meyvelerini vermeye devam etmektedir.",
                company="TestSirket", year=2023, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="interdepartmental",
            ),
            TestChunk(
                text="2023 mali yılında şirket 245 milyon TL net zarar açıklamıştır. Faaliyet zararı 180 milyon TL, finansman giderleri 420 milyon TL olarak gerçekleşmiştir.",
                company="TestSirket", year=2023, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="Bağımsız denetçi raporuna göre, şirketin 2023 yılı konsolide net zararı 245 milyon TL'dir.",
                company="TestSirket", year=2023, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1, 2],
        expected_removed=[0],
        description="Strateji: 'karlılık arttı' vs Finansal: 245M TL zarar. Pazarlama söylemi elenmeli.",
    ))

    # Test 2: Renewable energy investment claim
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="interdepartmental",
        question="Şirketin yenilenebilir enerji yatırımı ne kadar?",
        chunks=[
            TestChunk(
                text="Yeşil dönüşüm vizyonumuz kapsamında yenilenebilir enerjiye devasa yatırımlar yaparak sektörde lider konuma geldik. Enerji ihtiyacımızın büyük çoğunluğunu yenilenebilir kaynaklardan karşılıyoruz.",
                company="TestSirket", year=2023, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="interdepartmental",
            ),
            TestChunk(
                text="2023 yılında yenilenebilir enerji yatırımımız toplam 12 milyon TL olup, bu rakam toplam enerji harcamamızın %3'üne karşılık gelmektedir. Enerji ihtiyacımızın %94'ü doğalgaz ve kömürden karşılanmaktadır.",
                company="TestSirket", year=2023, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1],
        expected_removed=[0],
        description="'Devasa yatırım, lider konum' vs 'toplam %3, %94 fosil yakıt'. Greenwashing elenmeli.",
    ))

    # Test 3: Employee satisfaction
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="interdepartmental",
        question="Çalışan memnuniyeti durumu nedir?",
        chunks=[
            TestChunk(
                text="Çalışan bağlılığı anketimizde %92 memnuniyet oranına ulaştık. Şirketimiz, çalışanların en mutlu olduğu işyerleri arasında yer almaktadır.",
                company="TestSirket", year=2023, section_type="sosyal",
                reliability_weight=0.5, is_contradictory=True,
                contradiction_type="interdepartmental",
            ),
            TestChunk(
                text="2023 yılında personel devir hızı %38 olarak gerçekleşmiştir. 1.200 çalışan istifa etmiş, ortalama çalışma süresi 1.8 yıla gerilemiştir. İş mahkemesinde 45 aktif dava bulunmaktadır.",
                company="TestSirket", year=2023, section_type="yonetim",
                reliability_weight=0.7, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1],
        expected_removed=[0],
        description="%92 memnuniyet vs %38 devir hızı, 45 dava. Anket sonucu vs gerçek veriler çelişiyor.",
    ))

    return tests


def generate_gat_discriminating_tests() -> list[TestCase]:
    """
    Dimension 4: GAT-discriminating cases.

    These tests are designed so that BASIC MWIS (reliability * recency
    heuristic) makes the wrong call, but a temporally-aware,
    section-type-aware GAT can recover.

    Pattern: an OLD high-reliability chunk contradicts a NEW
    lower-reliability chunk. Pure MWIS keeps the old high-reliability
    one. Ground truth says the NEW one is correct (latest fact).
    Training GAT on these cases teaches it to weigh recency against
    reliability dynamically.
    """
    tests = []

    # GAT-1: Old finansal claim contradicted by newer cevre claim
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="gat_discriminating",
        question="Şirketin 2030 yenilenebilir enerji oranı hedefi nedir?",
        chunks=[
            TestChunk(
                text="2018 finansal raporumuzda 2030 yılına kadar yenilenebilir enerji payımızı %40'a çıkarma hedefi açıklanmıştır. Bu hedef, sermaye yatırım planlarımıza yansıtılmıştır.",
                company="TestSirket", year=2018, section_type="finansal",
                reliability_weight=0.9, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2024 sürdürülebilirlik raporumuzda yenilenebilir enerji hedefimiz revize edilerek 2030 itibarıyla %75 olarak güncellenmiştir. Yeni hedef, AB Yeşil Mutabakat çerçevesindedir.",
                company="TestSirket", year=2024, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1],
        expected_removed=[0],
        description="ESKI finansal (rel=0.9, 2018) vs YENI cevre (rel=0.6, 2024). "
                    "Pure MWIS reliability-baskın → idx 0 seçer (yanlış). "
                    "GAT temporal feature'larla idx 1'i tercih etmeli.",
    ))

    # GAT-2: Old strateji vs newer cevre — same direction
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="gat_discriminating",
        question="Şirketin biyoçeşitlilik koruma politikası nedir?",
        chunks=[
            TestChunk(
                text="2019 yönetim raporumuzda biyoçeşitlilik koruma için yıllık 5 milyon TL bütçe ayrılmıştır. Bu bütçe yönetim kurulu tarafından onaylanmıştır.",
                company="TestSirket", year=2019, section_type="yonetim",
                reliability_weight=0.7, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2024 itibarıyla biyoçeşitlilik koruma yatırımlarımız 32 milyon TL'ye yükseltilmiş olup yeni habitat restorasyon projeleri başlatılmıştır.",
                company="TestSirket", year=2024, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1],
        expected_removed=[0],
        description="2019 yönetim (rel=0.7) vs 2024 cevre (rel=0.6). MWIS yönetim'i seçebilir (rel daha yüksek). GAT recency'yi ağır basmalı.",
    ))

    # GAT-3: Recent finansal beats older finansal (within reliability tier)
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="gat_discriminating",
        question="Şirketin 2024 yılı net karı ne kadardır?",
        chunks=[
            TestChunk(
                text="2022 ara dönem mali tablomuzda 2024 yılı net kar projeksiyonu 850 milyon TL olarak yer almıştır.",
                company="TestSirket", year=2022, section_type="finansal",
                reliability_weight=0.9, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2024 yılsonu denetim raporuna göre net karımız 1.250 milyon TL olarak gerçekleşmiştir. Bağımsız denetimden geçmiştir.",
                company="TestSirket", year=2024, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1],
        expected_removed=[0],
        description="2022 projeksiyonu (850M) vs 2024 gerçeği (1250M). Aynı reliability, sadece recency belirleyici. MWIS recency formülü ile zaten doğru. GAT da aynı kararı vermeli.",
    ))

    # GAT-4: Cevre over yonetim with strong temporal contrast
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="gat_discriminating",
        question="Şirketin atık geri dönüşüm oranı nedir?",
        chunks=[
            TestChunk(
                text="2017 yönetim politikası dokümanında atık geri dönüşüm oranımızın %25'e ulaşması hedeflenmiştir. Bu rakam stratejik plan dahilinde belirlenmiştir.",
                company="TestSirket", year=2017, section_type="yonetim",
                reliability_weight=0.7, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2024 yılı çevre denetim raporumuzda atık geri dönüşüm oranımız %78 olarak ölçülmüş ve bağımsız denetçi tarafından doğrulanmıştır.",
                company="TestSirket", year=2024, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1],
        expected_removed=[0],
        description="2017 yönetim hedef (%25) vs 2024 cevre denetim (%78). 7 yıl fark, hedef çok eski. GAT recency'yi reliability'ye tercih etmeli.",
    ))

    # GAT-5: Three-way mix — old high-rel, mid-time mid-rel, new low-rel
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="gat_discriminating",
        question="Şirketin yıllık enerji tüketimi ne kadar?",
        chunks=[
            TestChunk(
                text="2018 finansal raporumuzda yıllık enerji tüketimimiz 450 GWh olarak raporlanmıştır.",
                company="TestSirket", year=2018, section_type="finansal",
                reliability_weight=0.9, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2024 çevre raporumuzda enerji tüketimimiz 290 GWh olarak ölçülmüş, enerji verimliliği yatırımları sonuç vermiştir.",
                company="TestSirket", year=2024, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="2024 sürdürülebilirlik özet raporumuza göre toplam enerji tüketimi 290 GWh civarındadır.",
                company="TestSirket", year=2024, section_type="strateji",
                reliability_weight=0.4, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1, 2],
        expected_removed=[0],
        description="2018 finansal (450GWh) eski. 2024 cevre+strateji yeni (290GWh). MWIS rel=0.9'u tercih edebilir. GAT recency'yi öğrenmeli.",
    ))

    return tests


def generate_cross_company_tests() -> list[TestCase]:
    """
    Dimension 5: Cross-Company Conflicts (NEW 2026-05-12).

    Aynı sektörden farklı iki şirket aynı topic'te zıt iddialarda bulunuyor.
    Bu, sistemin tek bir şirket içinde değil, çapraz-şirket düzeyinde de
    çelişki tespit edebildiğini doğrular. GAT'ın section_type + reliability
    feature'ları farklı şirketlerden gelen aynı tipte iddiaları
    karşılaştırma yeteneği kazandırmalı.
    """
    tests = []

    # CC-1: Banka A finansal denetimli vs Banka B strateji broşürü
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="cross_company",
        question="2023 yılı bankacılık sektörü karbon yoğunluğu hangi düzeyde?",
        chunks=[
            TestChunk(
                text="Bağımsız denetim raporumuza göre 2023 yılında toplam Scope 1+2 emisyonumuz 42.000 ton CO2e olup, kredi portföyümüzün karbon yoğunluğu sektör ortalamasının üzerindedir.",
                company="BankaA", year=2023, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="Sektörümüzde karbon emisyon yoğunluğu açısından lider konumdayız. 2023 itibarıyla portföyümüzün karbon ayak izi sektör ortalamasının çok altında kalmaktadır.",
                company="BankaB", year=2023, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="interdepartmental",
            ),
        ],
        expected_kept=[0],
        expected_removed=[1],
        description="BankaA denetimli rakam vs BankaB pazarlama söylemi. Reliability+section_type ile GAT denetimliyi tercih etmeli.",
    ))

    # CC-2: Same-sector, same-year, conflicting % claims
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="cross_company",
        question="Çimento sektöründe alternatif yakıt kullanım oranı nedir?",
        chunks=[
            TestChunk(
                text="2024 çevre raporumuzda alternatif yakıt kullanım oranımız %35 olarak doğrulanmıştır; bağımsız çevre denetiminden geçmiştir.",
                company="CimentoX", year=2024, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="2024 stratejik plan dokümanımızda alternatif yakıt kullanım oranımızın %78'e ulaştığı belirtilmektedir; sektör öncüsüyüz.",
                company="CimentoY", year=2024, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="scope",
            ),
        ],
        expected_kept=[0],
        expected_removed=[1],
        description="%35 denetimli (cevre) vs %78 stratejik iddia (strateji). 2.22x ratio + reliability farkı.",
    ))

    # CC-3: Three-way: two consistent, one outlier
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="cross_company",
        question="2024 perakende sektörü kadın yönetici oranı?",
        chunks=[
            TestChunk(
                text="2024 yılı insan kaynakları raporumuza göre kadın yönetici oranımız %42 olarak ölçülmüştür.",
                company="PerakendeA", year=2024, section_type="sosyal",
                reliability_weight=0.5, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="Bağımsız doğrulama sürecinden geçmiş kadın yönetici oranımız 2024 itibarıyla %38'dir.",
                company="PerakendeB", year=2024, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="Kadın yöneticilerimiz tüm üst kademede çoğunluğu oluşturmaktadır; sektörde tartışmasız liderliğimiz vardır.",
                company="PerakendeC", year=2024, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="interdepartmental",
            ),
        ],
        expected_kept=[0, 1],
        expected_removed=[2],
        description="İki sayısal değer (%42, %38) tutarlı, üçüncü vague strateji iddiası outlier.",
    ))

    return tests


def generate_zero_claim_tests() -> list[TestCase]:
    """
    Dimension 6: Zero-Claim Greenwashing (NEW 2026-05-12).

    "Sıfır atık", "tamamen yenilenebilir", "hiçbir emisyon" tipinde
    abartılı iddialar vs gerçek sayısal veriler. Bu testler özellikle
    extract_numbers'a eklenen Türkçe sıfır kelime tanıma için (gelecek)
    iyi bir benchmark olur.
    """
    tests = []

    # Z-1: "Sıfır atık" vs "45 ton"
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="zero_claim",
        question="Tesislerimizdeki tehlikeli atık miktarı?",
        chunks=[
            TestChunk(
                text="Sıfır atık vizyonumuz kapsamında 2024 yılında tüm operasyonel tesislerimizden hiçbir tehlikeli atık çıkmamıştır. Çevreye olan etkimiz tamamen ortadan kalkmıştır.",
                company="TestX", year=2024, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="interdepartmental",
            ),
            TestChunk(
                text="2024 yılı çevre denetiminde tesislerimizden toplam 45 ton tehlikeli atık raporlanmıştır. Bertaraf maliyetimiz 1.8 milyon TL'dir.",
                company="TestX", year=2024, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1],
        expected_removed=[0],
        description="'Sıfır' iddiası vs 45 ton gerçek. NLI semantic ya da Türkçe sıfır-kelime tanıması gerek.",
    ))

    # Z-2: "%100 yenilenebilir" vs scope-limited claim
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="zero_claim",
        question="Enerji ihtiyacının yenilenebilir karşılanma oranı?",
        chunks=[
            TestChunk(
                text="Şirket olarak enerji ihtiyacımızın tamamını yenilenebilir kaynaklardan karşıladığımızı gururla duyururuz. %100 yeşil enerji kullanımı sağlanmıştır.",
                company="TestY", year=2024, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="scope",
            ),
            TestChunk(
                text="2024 yılı denetimli enerji bilançomuzda yenilenebilir enerji payı %18 olarak hesaplanmıştır. Geri kalan %82'lik kısım fosil yakıt ağırlıklıdır.",
                company="TestY", year=2024, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1],
        expected_removed=[0],
        description="'%100 yeşil' iddiası vs %18 gerçek. Pazarlama vs denetimli veri çelişkisi.",
    ))

    return tests


def generate_full_testset() -> list[TestCase]:
    """Generate all test cases across 6 dimensions (was 4 before 2026-05-12)."""
    tests = []
    tests.extend(generate_temporal_tests())
    tests.extend(generate_scope_tests())
    tests.extend(generate_interdepartmental_tests())
    tests.extend(generate_gat_discriminating_tests())
    tests.extend(generate_cross_company_tests())     # NEW
    tests.extend(generate_zero_claim_tests())        # NEW
    return tests


def save_testset(tests: list[TestCase], path: Path = TESTSET_PATH):
    """Save test set to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for t in tests:
        d = {
            "test_id": t.test_id,
            "dimension": t.dimension,
            "question": t.question,
            "description": t.description,
            "expected_kept": t.expected_kept,
            "expected_removed": t.expected_removed,
            "chunks": [asdict(c) for c in t.chunks],
        }
        data.append(d)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(tests)} test cases to {path}")


def print_testset_summary(tests: list[TestCase]):
    """Print summary of test set."""
    from collections import Counter
    dims = Counter(t.dimension for t in tests)
    total_chunks = sum(len(t.chunks) for t in tests)
    contradictory = sum(
        sum(1 for c in t.chunks if c.is_contradictory) for t in tests
    )

    print(f"\n=== Sentetik Test Seti Özeti ===")
    print(f"Toplam test case:   {len(tests)}")
    print(f"Toplam chunk:       {total_chunks}")
    print(f"Çelişkili chunk:    {contradictory}")
    print(f"Temiz chunk:        {total_chunks - contradictory}")
    print(f"\nBoyut dağılımı:")
    for dim, cnt in dims.most_common():
        print(f"  {dim}: {cnt} test")


if __name__ == "__main__":
    tests = generate_full_testset()
    print_testset_summary(tests)
    save_testset(tests)

    print("\n=== Test Case Detayları ===")
    for t in tests:
        print(f"\n[{t.dimension}] {t.question}")
        print(f"  {t.description}")
        print(f"  Chunks: {len(t.chunks)}, Keep: {t.expected_kept}, Remove: {t.expected_removed}")
