"""
ReliabilityRAG Project Configuration
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT.parent / "ER MinerU Dosyaları"
CHUNKS_OUTPUT = PROJECT_ROOT / "data" / "chunks"
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vector_db"

# Chunking parameters
MAX_CHUNK_CHARS = 1500       # Max characters per chunk
MIN_CHUNK_CHARS = 100        # Skip chunks smaller than this
CHUNK_OVERLAP_CHARS = 200    # Overlap between consecutive chunks

# Section type classification keywords (Turkish)
SECTION_KEYWORDS = {
    "finansal": [
        "finansal", "gelir", "gider", "bilanço", "kar", "zarar", "nakit akış",
        "aktif", "pasif", "sermaye", "bütçe", "maliyet", "ciro", "faiz",
        "borç", "alacak", "özkaynak", "temettü", "hisse", "yatırım",
        "konsolide", "denetim", "bağımsız denetçi", "tablo", "milyon tl",
        "bin tl", "ekonomik değer", "ebitda", "satış gelirleri",
    ],
    "cevre": [
        "çevre", "emisyon", "karbon", "iklim", "enerji", "su tüketim",
        "atık", "geri dönüşüm", "biyoçeşitlilik", "sera gazı", "co2",
        "yenilenebilir", "sürdürülebilir", "çevresel", "ekolojik",
        "karbon ayak izi", "ghg", "scope", "kapsam",
    ],
    "sosyal": [
        "çalışan", "istihdam", "iş sağlığı", "güvenliği", "eğitim",
        "toplum", "sosyal sorumluluk", "insan hakları", "fırsat eşitliği",
        "kadın", "çeşitlilik", "kapsayıcılık", "gönüllü", "bağış",
        "isg", "kaza", "meslek hastalığı",
    ],
    "yonetim": [
        "yönetim kurulu", "kurumsal yönetim", "etik", "uyum", "risk",
        "iç denetim", "komite", "bağımsız üye", "genel kurul",
        "organizasyon", "şeffaflık", "hesap verebilirlik", "politika",
    ],
    "strateji": [
        "strateji", "vizyon", "misyon", "hedef", "plan", "büyüme",
        "dönüşüm", "inovasyon", "dijital", "ar-ge", "araştırma",
        "gelecek", "roadmap", "değer yaratma", "iş modeli",
    ],
}

# Reliability weights by section type
# Financial sections (audited) get highest weight, marketing/strategy lowest
SECTION_RELIABILITY_WEIGHTS = {
    "finansal": 0.9,
    "cevre": 0.6,
    "sosyal": 0.5,
    "yonetim": 0.7,
    "strateji": 0.4,
    "genel": 0.5,  # default / unclassified
}


# ── GPU Profile Configuration ───────────────────────────────────────────
# Auto-selects models + batch sizes based on available VRAM.
# Override with env var: RELIABILITY_RAG_PROFILE=small|medium|large
#
# NOTE: EMBEDDING_MODEL is intentionally NOT in profiles — changing it would
# invalidate the saved .npy file (182K × 768-dim). Keep e5-base everywhere.

GPU_PROFILES = {
    # RTX 2060 6GB, GTX 1660, low-VRAM laptops
    "small": {
        "llm_model": "microsoft/Phi-3.5-mini-instruct",
        "llm_dtype": "4bit",          # bitsandbytes quantization
        "nli_model": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "nli_batch_size": 32,
        "nli_truncation_chars": 256,
        "isolated_answer_max_tokens": 150,
        "final_answer_max_tokens": 600,
    },
    # RTX 3060 12GB, RTX 4070 12GB
    "medium": {
        "llm_model": "microsoft/Phi-3.5-mini-instruct",
        "llm_dtype": "fp16",
        "nli_model": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "nli_batch_size": 64,
        "nli_truncation_chars": 512,
        "isolated_answer_max_tokens": 200,
        "final_answer_max_tokens": 800,
    },
    # RTX 5080 16GB, RTX 4080 16GB — Akvaryum PC
    "large": {
        "llm_model": "Qwen/Qwen2.5-7B-Instruct",
        "llm_dtype": "4bit",
        "nli_model": "MoritzLaurer/mDeBERTa-v3-large-mnli-xnli",
        "nli_batch_size": 128,
        "nli_truncation_chars": 1024,
        "isolated_answer_max_tokens": 300,
        "final_answer_max_tokens": 1200,
    },
    # RTX 3090/4090 24GB, A6000, H100 — full-fat
    "xlarge": {
        "llm_model": "Qwen/Qwen2.5-14B-Instruct",
        "llm_dtype": "fp16",
        "nli_model": "MoritzLaurer/mDeBERTa-v3-large-mnli-xnli",
        "nli_batch_size": 256,
        "nli_truncation_chars": 1500,
        "isolated_answer_max_tokens": 400,
        "final_answer_max_tokens": 2000,
    },
}


def get_profile(override: str = None) -> dict:
    """
    Auto-select GPU profile based on available VRAM (or env var).

    Thresholds:
      <  8 GB → small     (RTX 2060)
      <  14 GB → medium    (RTX 3060/4070)
      <  20 GB → large    (RTX 5080/4080)   ← Akvaryum PC
      >= 20 GB → xlarge   (RTX 3090/4090)
    """
    import os

    name = override or os.environ.get("RELIABILITY_RAG_PROFILE")
    if name:
        if name not in GPU_PROFILES:
            raise ValueError(f"Unknown profile '{name}'. Valid: {list(GPU_PROFILES)}")
        return {"_name": name, **GPU_PROFILES[name]}

    try:
        import torch
        if not torch.cuda.is_available():
            name = "small"
        else:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb < 8:
                name = "small"
            elif vram_gb < 14:
                name = "medium"
            elif vram_gb < 20:
                name = "large"
            else:
                name = "xlarge"
    except Exception:
        name = "small"

    return {"_name": name, **GPU_PROFILES[name]}
