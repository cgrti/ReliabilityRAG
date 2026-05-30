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

    # GAT-6: Kadın istihdamı — eski finansal (yüksek rel) vs yeni sosyal (düşük rel)
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="gat_discriminating",
        question="Şirketin kadın çalışan oranı nedir?",
        chunks=[
            TestChunk(
                text="2019 finansal raporumuzda kadın çalışan oranımız %18 olarak raporlanmış olup, bağımsız denetim sürecinden geçmiştir.",
                company="TestSirket", year=2019, section_type="finansal",
                reliability_weight=0.9, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2024 yılı insan kaynakları raporumuza göre kadın çalışan oranımız %42'ye yükselmiştir. Çeşitlilik politikamızın somut sonucudur.",
                company="TestSirket", year=2024, section_type="sosyal",
                reliability_weight=0.5, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1],
        expected_removed=[0],
        description="2019 finansal (rel=0.9) %18 ESKI vs 2024 sosyal (rel=0.5) %42 YENI. "
                    "Reliability farkı 0.4, 5 yıl recency farkı. MWIS reliability-baskın → eski seçer (yanlış). "
                    "GAT recency feature'ı ile 2024'ü tercih etmeli.",
    ))

    # GAT-7: Dijital dönüşüm budget — yönetim (orta-rel) vs strateji (düşük-rel), büyük ölçek farkı
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="gat_discriminating",
        question="Şirketin dijital dönüşüm yatırımı ne kadar?",
        chunks=[
            TestChunk(
                text="2018 yönetim kurulu kararıyla dijital dönüşüm bütçesi yıllık 8 milyon TL olarak onaylanmıştır.",
                company="TestSirket", year=2018, section_type="yonetim",
                reliability_weight=0.7, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2024 stratejik plan dokümanımızda dijital dönüşüm yatırımımız 95 milyon TL'ye çıkarılmış olup, ana iş kollarımızın dijitalleştirilmesini hedeflemektedir.",
                company="TestSirket", year=2024, section_type="strateji",
                reliability_weight=0.4, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1],
        expected_removed=[0],
        description="2018 yönetim (rel=0.7) 8M vs 2024 strateji (rel=0.4) 95M. "
                    "11.8x büyüme — eski rakam güncel olamaz. MWIS yönetim'i tercih edebilir; "
                    "GAT temporal-aware kararla 2024'ü seçmeli.",
    ))

    # GAT-8: Emisyon yoğunluğu — eski finansal (yüksek rel) vs yeni cevre (orta rel), 6 yıl fark
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="gat_discriminating",
        question="Şirketin birim üretim başına karbon emisyon yoğunluğu nedir?",
        chunks=[
            TestChunk(
                text="2017 yılı denetimli finansal raporumuza göre birim üretim başına karbon emisyonumuz 0.85 ton CO2e/ton ürün olarak hesaplanmıştır.",
                company="TestSirket", year=2017, section_type="finansal",
                reliability_weight=0.9, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2023 sürdürülebilirlik raporumuzda emisyon yoğunluğumuz 0.42 ton CO2e/ton ürün'e düşürülmüştür. Enerji verimliliği projelerinin somut sonucudur.",
                company="TestSirket", year=2023, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1],
        expected_removed=[0],
        description="2017 finansal (rel=0.9) 0.85 vs 2023 cevre (rel=0.6) 0.42. "
                    "2.0x ratio + 6 yıl fark. MWIS reliability-baskın → eski seçer.",
    ))

    # GAT-9: 4-chunk dense temporal evolution
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="gat_discriminating",
        question="Şirketin güncel yenilenebilir enerji oranı nedir?",
        chunks=[
            TestChunk(
                text="2017 finansal raporumuzda yenilenebilir enerji oranımız %8 olarak belirtilmiş, sermaye yatırım planına dahil edilmiştir.",
                company="TestSirket", year=2017, section_type="finansal",
                reliability_weight=0.9, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2020 yönetim kurulu raporumuzda yenilenebilir enerji oranı %22'ye yükseltilmiştir.",
                company="TestSirket", year=2020, section_type="yonetim",
                reliability_weight=0.7, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2024 yılı çevre raporumuza göre yenilenebilir enerji oranımız %55'e ulaşmıştır.",
                company="TestSirket", year=2024, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="2024 stratejik plan dokümanımızda yenilenebilir enerji payımız %55 olarak raporlanmıştır.",
                company="TestSirket", year=2024, section_type="strateji",
                reliability_weight=0.4, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[2, 3],
        expected_removed=[0, 1],
        description="4 chunk, 3 yıl: 2017 (rel=0.9, %8), 2020 (rel=0.7, %22), 2024 cevre (rel=0.6, %55), 2024 strateji (rel=0.4, %55). "
                    "Sadece son ikisi tutarlı ve güncel. MWIS yüksek-rel eski'leri tercih edebilir; "
                    "GAT temporal kümeleme ile 2024'leri seçmeli.",
    ))

    # GAT-10: Same-reliability tier — pure recency (regression guard, kolay test)
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="gat_discriminating",
        question="Şirketin son denetimli net karı ne kadar?",
        chunks=[
            TestChunk(
                text="2020 yılsonu denetim raporumuza göre net karımız 480 milyon TL olarak gerçekleşmiştir.",
                company="TestSirket", year=2020, section_type="finansal",
                reliability_weight=0.9, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2023 yılsonu denetim raporumuza göre net karımız 1.350 milyon TL olarak gerçekleşmiştir.",
                company="TestSirket", year=2023, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[1],
        expected_removed=[0],
        description="Aynı reliability (rel=0.9), 3 yıl fark. Pure recency belirleyici. "
                    "MWIS recency formülü ile zaten doğru bekleniyor — GAT da regresyon yapmamalı.",
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


def generate_dense_graph_tests() -> list[TestCase]:
    """
    Dimension 7: Dense Contradiction Graphs (NEW 2026-05-30).

    8 chunk'lık çok-yollu çelişki ağları. GAT'ın attention mechanism'inin
    çoklu kontekst sinyallerini birleştirme yeteneğini test eder. Pure MWIS
    bu kadar yoğun çelişki yapılarında alt-optimal subset seçebilir; GAT'ın
    learned weighting'i daha tutarlı sonuçlar vermeli.

    Bu testler ablation istatistiksel gücünü artırmak için kritik:
    küçük graf'larda MWIS≈GAT, büyük graf'larda fark açılmalı.
    """
    tests = []

    # DG-1: Karbon emisyon yoğunluğu — 8 chunk, 3 şirket, 4 yıl
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="dense_graph",
        question="Çimento sektöründe karbon emisyon yoğunluğu (ton CO2e/ton ürün) hangi düzeydedir?",
        chunks=[
            TestChunk(
                text="2019 yılı çevre raporumuzda emisyon yoğunluğumuz 0.78 ton CO2e/ton olarak raporlanmıştır.",
                company="CimA", year=2019, section_type="cevre",
                reliability_weight=0.6, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2024 bağımsız denetim raporumuza göre emisyon yoğunluğumuz 0.52 ton CO2e/ton'a düşürülmüştür.",
                company="CimA", year=2024, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="2024 stratejik plan dokümanımızda emisyon yoğunluğumuz 0.30 ton CO2e/ton olarak öne çıkartılmaktadır.",
                company="CimA", year=2024, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="interdepartmental",
            ),
            TestChunk(
                text="2023 yılı çevre raporumuzda emisyon yoğunluğumuz 0.58 ton CO2e/ton olarak gerçekleşmiştir.",
                company="CimB", year=2023, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="2024 yönetim raporumuzda emisyon yoğunluğumuz 0.55 ton CO2e/ton'a iyileştirilmiştir.",
                company="CimB", year=2024, section_type="yonetim",
                reliability_weight=0.7, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="2018 yıl raporumuzda emisyon yoğunluğumuz 0.95 ton CO2e/ton düzeyinde idi.",
                company="CimB", year=2018, section_type="yonetim",
                reliability_weight=0.7, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2024 sürdürülebilirlik raporumuzda emisyon yoğunluğumuz 0.61 ton CO2e/ton olarak ölçülmüştür.",
                company="CimC", year=2024, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="Karbon ayak izinde sektör lideri konumdayız; emisyon yoğunluğumuz sektör ortalamasının çok altında kalmaktadır.",
                company="CimC", year=2024, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="interdepartmental",
            ),
        ],
        expected_kept=[1, 3, 4, 6],
        expected_removed=[0, 2, 5, 7],
        description="8 chunk: 3 şirket × birden çok yıl. Tutarlı (2023-2024 cevre/finansal/yonetim 0.52-0.61 arası), "
                    "eski (2018-2019: 0.78-0.95) elenmeli, strateji abartıları (0.30, vague liderlik) elenmeli. "
                    "Dense graph: ≥5 çelişki edge. MWIS subset-optimal seçemeyebilir.",
    ))

    # DG-2: Çalışan sayısı temporal evolution + 1 outlier
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="dense_graph",
        question="Şirketin güncel çalışan sayısı nedir?",
        chunks=[
            TestChunk(
                text="2019 yıl sonunda toplam çalışan sayımız 14.200 olarak raporlanmıştır.",
                company="TestSirket", year=2019, section_type="sosyal",
                reliability_weight=0.5, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2020 yılı insan kaynakları raporumuza göre çalışan sayımız 13.500 düzeyindedir.",
                company="TestSirket", year=2020, section_type="sosyal",
                reliability_weight=0.5, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2021 itibarıyla çalışan sayımız 12.800 olarak gerçekleşmiştir.",
                company="TestSirket", year=2021, section_type="yonetim",
                reliability_weight=0.7, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2022 yılı finansal denetim raporumuzda 11.450 aktif çalışanımız bulunmaktadır.",
                company="TestSirket", year=2022, section_type="finansal",
                reliability_weight=0.9, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2023 yılı sosyal sorumluluk raporumuza göre çalışan sayımız 10.800'e düşmüştür.",
                company="TestSirket", year=2023, section_type="sosyal",
                reliability_weight=0.5, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2024 yılsonu denetimli finansal raporumuza göre çalışan sayımız 10.250 düzeyindedir.",
                company="TestSirket", year=2024, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="2024 yılı insan kaynakları raporumuza göre 10.200 çalışanımız mevcuttur.",
                company="TestSirket", year=2024, section_type="sosyal",
                reliability_weight=0.5, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="Stratejik vizyonumuza göre çalışan sayımız 25.000'e ulaşmıştır; sektörde lider istihdam sağlayıcı konumdayız.",
                company="TestSirket", year=2024, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="interdepartmental",
            ),
        ],
        expected_kept=[5, 6],
        expected_removed=[0, 1, 2, 3, 4, 7],
        description="8 chunk: 5 yıllık azalan temporal seri (14.2K→10.2K) + 2024'te 1 outlier (25K abartı). "
                    "Sorgu 'güncel' istiyor → sadece 2024 chunk'ları doğru (5,6). "
                    "MWIS 2022 finansal (rel=0.9) eski chunk'ı tercih ederek hata yapabilir; "
                    "GAT temporal + interdepartmental filtreyi birleştirip 2024 finansal+sosyal'ı seçmeli.",
    ))

    # DG-3: Sürdürülebilirlik yatırımı — 8 chunk, 4 şirket
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="dense_graph",
        question="Şirketlerin 2024 yılı sürdürülebilirlik yatırımı ne kadar?",
        chunks=[
            TestChunk(
                text="2024 finansal raporumuzda sürdürülebilirlik yatırımımız 145 milyon TL olarak doğrulanmıştır.",
                company="SektorA", year=2024, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="Sürdürülebilirlik yatırımı kapsamında devasa kaynaklar ayırdık; sektörde en büyük yatırımcıyız.",
                company="SektorA", year=2024, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="interdepartmental",
            ),
            TestChunk(
                text="2024 yılı bağımsız denetim raporumuza göre sürdürülebilirlik yatırımımız 88 milyon TL düzeyindedir.",
                company="SektorB", year=2024, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="2022 finansal raporumuzda 2024 yılı sürdürülebilirlik yatırım projeksiyonumuz 220 milyon TL idi.",
                company="SektorB", year=2022, section_type="finansal",
                reliability_weight=0.9, is_contradictory=True,
                contradiction_type="temporal",
            ),
            TestChunk(
                text="2024 yılı çevre raporumuzda sürdürülebilirlik yatırımımız 62 milyon TL olarak hesaplanmıştır.",
                company="SektorC", year=2024, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="Sürdürülebilirlik vizyonumuz sınırsız; her yıl daha fazla yatırım yapıyoruz.",
                company="SektorC", year=2024, section_type="strateji",
                reliability_weight=0.4, is_contradictory=True,
                contradiction_type="interdepartmental",
            ),
            TestChunk(
                text="2024 yönetim kurulu raporumuzda sürdürülebilirlik harcamamız 71 milyon TL olarak onaylanmıştır.",
                company="SektorD", year=2024, section_type="yonetim",
                reliability_weight=0.7, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="2024 finansal raporumuzda sürdürülebilirlik yatırımımız 73 milyon TL olarak doğrulanmıştır.",
                company="SektorD", year=2024, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[0, 2, 4, 6, 7],
        expected_removed=[1, 3, 5],
        description="8 chunk, 4 şirket. Her şirketin sayısal doğrulanmış değeri var; "
                    "abartı strateji iddiaları (1, 5) ve eski projeksiyon (3) elenmeli. "
                    "SektorD'de 71M (yonetim) vs 73M (finansal) yakın değerler — çelişki SAYILMAMALI (numerical_edge tetiklemeli).",
    ))

    return tests


def generate_numerical_edge_tests() -> list[TestCase]:
    """
    Dimension 8: Numerical False-Positive Guards (NEW 2026-05-30).

    Sistemin sayısal çelişki tespitinin PRECISION'ını test eden NEGATIF
    testler. Bu chunk'lar arasında gerçek çelişki YOKTUR; sistem çelişki
    olarak işaretlerse FALSE POSITIVE üretmiş olur.

    extract_numbers + numerical_conflict_score'un yakın değerleri, farklı
    birimleri, farklı scope'ları çelişki sanmamasını test eder. Recall %67'yi
    geçtikten sonra precision tarafını da ölçmemiz lazım — F1 için.
    """
    tests = []

    # NE-1: Aynı metrik ufak fark (rounding/methodology tolerance)
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="numerical_edge",
        question="Şirketin yıllık enerji tüketimi ne kadar?",
        chunks=[
            TestChunk(
                text="2024 yılı çevre raporumuza göre toplam enerji tüketimimiz 285.5 GWh olarak ölçülmüştür.",
                company="TestSirket", year=2024, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="2024 yılsonu denetim raporumuzda enerji tüketimimiz 290 GWh olarak yuvarlanmış değerle raporlanmıştır.",
                company="TestSirket", year=2024, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[0, 1],
        expected_removed=[],
        description="285.5 vs 290 GWh — %1.6 fark, methodology rounding. Çelişki DEĞİL. "
                    "Sistem her ikisini de tutmalı; numerical_conflict_score'un MIN-ratio mantığı "
                    "(ratio 1.016) bu tolerance içinde kalmalı, false-positive üretmemeli.",
    ))

    # NE-2: Farklı birim aynı değer — currency/scale ambiguity
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="numerical_edge",
        question="Şirketin 2024 yılı toplam yatırım miktarı?",
        chunks=[
            TestChunk(
                text="2024 yılı yatırım harcamalarımız 8.2 milyar TL düzeyinde gerçekleşmiştir.",
                company="TestSirket", year=2024, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="2024 yatırımlarımız 8.200 milyon TL'ye ulaşmıştır.",
                company="TestSirket", year=2024, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[0, 1],
        expected_removed=[],
        description="8.2 milyar TL = 8.200 milyon TL — aynı miktar, farklı birim. "
                    "Naif numerical_conflict (8.2 vs 8200 = 1000x ratio) tetikleyebilir; "
                    "NLI semantic match olur ya da birim normalization gerek. "
                    "Bu test bilinen bir gap: ileride birim normalization eklenirse fail→pass.",
    ))

    # NE-3: Farklı scope, farklı sayı — scope farkı, çelişki değil
    tests.append(TestCase(
        test_id=str(uuid.uuid4()),
        dimension="numerical_edge",
        question="Şirketin sera gazı emisyonu ne kadardır?",
        chunks=[
            TestChunk(
                text="Kapsam 1 (doğrudan) emisyonlarımız 2024 yılında 95 ton CO2e olarak ölçülmüştür.",
                company="TestSirket", year=2024, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="Kapsam 1+2 toplam emisyonumuz 2024 yılında 240 ton CO2e düzeyindedir.",
                company="TestSirket", year=2024, section_type="cevre",
                reliability_weight=0.6, is_contradictory=False,
                contradiction_type="none",
            ),
            TestChunk(
                text="Kapsam 1+2+3 dahil toplam emisyonumuz 2024'te 1.150 ton CO2e olarak hesaplanmıştır; bağımsız doğrulamadan geçmiştir.",
                company="TestSirket", year=2024, section_type="finansal",
                reliability_weight=0.9, is_contradictory=False,
                contradiction_type="none",
            ),
        ],
        expected_kept=[0, 1, 2],
        expected_removed=[],
        description="95 / 240 / 1150 ton — büyük sayısal farklar VAR ama scope farklı (Kapsam 1, 1+2, 1+2+3). "
                    "Bunlar aynı metric değil. Hiçbiri çelişki olarak işaretlenmemeli. "
                    "NLI semantik olarak 'Kapsam' kelimesini tanıyıp farkı görmeli; "
                    "tetiklerse contradiction graph false-edge üretiyor demektir.",
    ))

    return tests


def generate_full_testset() -> list[TestCase]:
    """Generate all test cases across 8 dimensions (was 6 before 2026-05-30)."""
    tests = []
    tests.extend(generate_temporal_tests())
    tests.extend(generate_scope_tests())
    tests.extend(generate_interdepartmental_tests())
    tests.extend(generate_gat_discriminating_tests())
    tests.extend(generate_cross_company_tests())
    tests.extend(generate_zero_claim_tests())
    tests.extend(generate_dense_graph_tests())       # NEW 2026-05-30
    tests.extend(generate_numerical_edge_tests())    # NEW 2026-05-30
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
    # Windows cp1254 console can't encode Turkish punctuation (→, —, etc.)
    # used in test descriptions. Reconfigure stdout to UTF-8 with replace
    # fallback so detail print loop doesn't crash on Windows.
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7

    tests = generate_full_testset()
    print_testset_summary(tests)
    save_testset(tests)

    print("\n=== Test Case Detayları ===")
    for t in tests:
        print(f"\n[{t.dimension}] {t.question}")
        print(f"  {t.description}")
        print(f"  Chunks: {len(t.chunks)}, Keep: {t.expected_kept}, Remove: {t.expected_removed}")
