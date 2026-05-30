"""
LLM Engine for ReliabilityRAG

Supports multiple backends:
1. Transformers (local, default) - Phi-3-mini or Gemma-2 2B for 6GB VRAM
2. Ollama (local, optional) - if installed
3. API (remote, optional) - OpenAI-compatible endpoints

All backends implement the same interface: generate(prompt) -> str
"""
import os
from abc import ABC, abstractmethod
from typing import Optional


class LLMEngine(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        pass

    def generate_stream(self, prompt: str, max_tokens: int = 512):
        """
        Stream generation — yields cumulative decoded text as tokens arrive.
        Default implementation falls back to blocking generate() + single yield.
        Subclasses can override for true token-by-token streaming.
        """
        yield self.generate(prompt, max_tokens=max_tokens)


class TransformersEngine(LLMEngine):
    """
    Local LLM via HuggingFace transformers.

    Model + dtype come from config.GPU_PROFILES (auto-selected by VRAM).
    Override with env var RELIABILITY_RAG_PROFILE or model_name argument.
    """

    def __init__(self, model_name: str = None, dtype: str = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from config import get_profile

        profile = get_profile()
        model_name = model_name or profile["llm_model"]
        dtype = dtype or profile["llm_dtype"]

        print(f"GPU profile: {profile['_name']}")
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU VRAM: {vram_gb:.1f} GB")
        print(f"Loading LLM: {model_name} (dtype={dtype})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if dtype == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        elif dtype == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        elif dtype == "fp16":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        elif dtype == "bf16":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            raise ValueError(f"Unknown dtype '{dtype}'. Use 4bit|fp16|bf16")

        self.model.eval()
        print("LLM loaded successfully!")

    def _build_inputs(self, prompt: str):
        """Shared tokenization path for generate() and generate_stream()."""
        messages = [{"role": "user", "content": prompt}]
        # Two-step tokenization for cross-version compatibility (transformers 4.x and 5.x):
        # apply_chat_template(return_tensors=...) behaves inconsistently across versions,
        # so we render to a string first and tokenize explicitly.
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        return inputs

    def _get_terminators(self):
        """
        Stop-token list for model.generate(). Llama-3 family (incl. Turkish-Llama)
        REQUIRES <|eot_id|> (128009) as a terminator: their tokenizer.eos_token_id
        is <|end_of_text|> (document-end, 128001), not <|eot_id|> (turn-end).
        Without this the model emits <|eot_id|> at turn-end, generation continues
        past it, and skip_special_tokens decode strips <|...|> tags but leaks the
        role-header words ("assistant"/"user"/"system") as plain text → the
        fake multi-turn hallucination observed on Turkish-Llama-8b 2026-05-30.
        Safe for Qwen too: its tokenizer doesn't have <|eot_id|> so the lookup
        returns unk and we skip it.
        """
        terms = [self.tokenizer.eos_token_id]
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None and eot != self.tokenizer.unk_token_id:
            terms.append(eot)
        return terms

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        import torch

        inputs = self._build_inputs(prompt)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                # Repetition guard: smoke test 2026-04-18 showed a 42-min
                # runaway on a Cimento query where the model hit a loop and
                # never emitted EOS. These two bound loops aggressively
                # without hurting fluency for Turkish outputs.
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                eos_token_id=self._get_terminators(),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = output[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_stream(self, prompt: str, max_tokens: int = 512):
        """
        Token-by-token streaming via TextIteratorStreamer.
        Yields cumulative text (not deltas) so UI can replace the whole field
        on each update. Runs model.generate() on a background thread.
        """
        import threading
        from transformers import TextIteratorStreamer

        inputs = self._build_inputs(prompt)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            # Match generate(): prevent 40-min runaway loops.
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            eos_token_id=self._get_terminators(),
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Run generation on a worker thread so we can consume the streamer.
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        accumulated = ""
        for chunk in streamer:
            accumulated += chunk
            yield accumulated
        thread.join()


class OllamaEngine(LLMEngine):
    """Local LLM via Ollama REST API."""

    def __init__(self, model: str = "phi3:mini", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        print(f"Using Ollama: {model} at {base_url}")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        import requests
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


class APIEngine(LLMEngine):
    """Remote LLM via OpenAI-compatible API."""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.model = model
        print(f"Using API: {model} at {self.base_url}")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        import requests
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ── Prompt Templates ────────────────────────────────────────────────────

ISOLATED_ANSWER_PROMPT = """Aşağıdaki bağlam metnine dayanarak soruyu kısaca yanıtla.
SADECE verilen metne göre cevap ver, dışarıdan bilgi ekleme.
Eğer metinde soruyla ilgili bilgi yoksa "Bu metinde ilgili bilgi bulunmuyor" de.

Bağlam [{company} {year} - {section_type}]:
{context}

Soru: {question}

Cevap:"""

FINAL_ANSWER_PROMPT = """Sen güvenilir bir kurumsal rapor analisti asistanısın.
Aşağıdaki kaynaklara dayanarak soruyu kapsamlı ve doğru bir şekilde yanıtla.

Kurallar:
- SADECE verilen kaynaklara dayanarak cevap ver
- Kaynaklar arasında çelişki varsa bunu belirt
- Yıl bilgisini vererek hangi döneme ait olduğunu açıkça söyle
- Sayısal verileri doğru aktar
- Emin olmadığın bilgileri tahmin olarak belirt
- Şirket adlarını ve YILLARI kaynaklarda yazıldığı gibi BİREBİR aktar; asla değiştirme veya uydurma

Kaynaklar:
{sources}

Soru: {question}

Cevap:"""


def get_engine(backend: str = "transformers", **kwargs) -> LLMEngine:
    """Factory function to create the appropriate LLM engine."""
    if backend == "transformers":
        return TransformersEngine(**kwargs)
    elif backend == "ollama":
        return OllamaEngine(**kwargs)
    elif backend == "api":
        return APIEngine(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
