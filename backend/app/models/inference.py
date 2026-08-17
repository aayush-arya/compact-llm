"""
Loads the base model + LoRA adapter once at process startup and exposes a
single generate() function. Both /score and /compare call into the same
loaded weights -- the adapter is toggled on/off via PEFT's
`disable_adapter()` context manager rather than reloading anything, so a
/compare call costs one extra forward pass, not one extra model load.

Three backends, selected by settings.model_backend:
  - "transformers": real inference via transformers + peft + bitsandbytes.
    Needs a GPU (or a lot of patience on CPU) and the trained adapter.
  - "ollama": calls a local/remote Ollama server serving the GGUF export
    (see training/export_gguf.py). CPU-friendly, good for cheap hosting.
  - "mock": deterministic fake scorer with no model weights required.
    Default, so the API/frontend are runnable without a GPU or trained
    checkpoint for local development.
"""
import hashlib
import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

import httpx

from app.config import get_settings

INSTRUCTION = (
    "Score how well this resume matches this job description (0-100) and "
    "explain why in 2 sentences."
)


@dataclass
class GenerationResult:
    score: int
    rationale: str
    latency_ms: float


def _build_prompt(resume: str, jd: str) -> str:
    return f"{INSTRUCTION}\n\nResume: {resume}\n\nJob Description: {jd}"


def parse_output(text: str) -> tuple[int, str]:
    m = re.search(r"Score:\s*(\d{1,3})", text)
    r = re.search(r"Rationale:\s*(.+)", text, re.DOTALL)
    score = max(0, min(100, int(m.group(1)))) if m else 50
    rationale = " ".join(r.group(1).strip().split()) if r else text.strip()[:400]
    return score, rationale


class InferenceEngine:
    def generate(self, resume: str, jd: str, use_adapter: bool) -> GenerationResult:
        raise NotImplementedError

    def generate_stream(self, resume: str, jd: str, use_adapter: bool) -> Iterator[str]:
        """Default: no real token streaming available, yield the full text in one chunk."""
        result = self.generate(resume, jd, use_adapter)
        yield f"Score: {result.score}\nRationale: {result.rationale}"


class MockEngine(InferenceEngine):
    """Deterministic, hash-based fake scorer so the full stack is demoable
    without a GPU or trained weights. The fine-tuned path is nudged to look
    modestly better/faster than the base path so /compare has something
    visible to show -- replace with a real backend before benchmarking."""

    def generate(self, resume: str, jd: str, use_adapter: bool) -> GenerationResult:
        start = time.perf_counter()
        digest = hashlib.sha256((resume + jd).encode()).hexdigest()
        base_score = int(digest[:4], 16) % 101
        overlap = len(set(resume.lower().split()) & set(jd.lower().split()))
        score = min(100, base_score // 3 + overlap * 2) if use_adapter else base_score
        rationale = (
            "Mock response: overlap between resume and JD terms drives this score "
            "-- swap MODEL_BACKEND for `transformers` or `ollama` for a real result."
        )
        # simulate the fine-tuned small model being faster than the base + longer prompt path
        time.sleep(0.15 if use_adapter else 0.35)
        latency_ms = (time.perf_counter() - start) * 1000
        return GenerationResult(score=score, rationale=rationale, latency_ms=latency_ms)

    def generate_stream(self, resume: str, jd: str, use_adapter: bool) -> Iterator[str]:
        digest = hashlib.sha256((resume + jd).encode()).hexdigest()
        base_score = int(digest[:4], 16) % 101
        overlap = len(set(resume.lower().split()) & set(jd.lower().split()))
        score = min(100, base_score // 3 + overlap * 2) if use_adapter else base_score
        full_text = (
            f"Score: {score}\nRationale: Mock response: overlap between resume and JD terms "
            "drives this score -- swap MODEL_BACKEND for `transformers` or `ollama` for a real result."
        )
        for word in full_text.split(" "):
            time.sleep(0.02)
            yield word + " "


class TransformersEngine(InferenceEngine):
    def __init__(self, base_model_path: str, adapter_path: str):
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_path, max_seq_length=2048, dtype=None, load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, adapter_path)

    def generate(self, resume: str, jd: str, use_adapter: bool) -> GenerationResult:
        import torch

        prompt = _build_prompt(resume, jd)
        chat = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        inputs = self.tokenizer([chat], return_tensors="pt").to(self.model.device)

        start = time.perf_counter()
        ctx = self.model.disable_adapter() if not use_adapter else _noop_ctx()
        with ctx:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=150, use_cache=True, do_sample=False,
                )
        latency_ms = (time.perf_counter() - start) * 1000

        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        score, rationale = parse_output(text)
        return GenerationResult(score=score, rationale=rationale, latency_ms=latency_ms)

    def generate_stream(self, resume: str, jd: str, use_adapter: bool) -> Iterator[str]:
        import threading

        from transformers import TextIteratorStreamer

        prompt = _build_prompt(resume, jd)
        chat = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        inputs = self.tokenizer([chat], return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        ctx = self.model.disable_adapter() if not use_adapter else _noop_ctx()
        with ctx:
            kwargs = dict(**inputs, max_new_tokens=150, use_cache=True, do_sample=False, streamer=streamer)
            thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
            thread.start()
            for chunk in streamer:
                yield chunk
            thread.join()


class _noop_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class OllamaEngine(InferenceEngine):
    def __init__(self, host: str, finetuned_model: str, base_model: str):
        self.host = host.rstrip("/")
        self.finetuned_model = finetuned_model
        self.base_model = base_model
        self.client = httpx.Client(timeout=120)

    def generate(self, resume: str, jd: str, use_adapter: bool) -> GenerationResult:
        prompt = _build_prompt(resume, jd)
        model = self.finetuned_model if use_adapter else self.base_model

        start = time.perf_counter()
        resp = self.client.post(
            f"{self.host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000

        text = resp.json().get("response", "")
        score, rationale = parse_output(text)
        return GenerationResult(score=score, rationale=rationale, latency_ms=latency_ms)

    def generate_stream(self, resume: str, jd: str, use_adapter: bool) -> Iterator[str]:
        prompt = _build_prompt(resume, jd)
        model = self.finetuned_model if use_adapter else self.base_model
        with self.client.stream(
            "POST", f"{self.host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": True},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("response"):
                    yield data["response"]
                if data.get("done"):
                    break


@lru_cache
def get_engine() -> InferenceEngine:
    settings = get_settings()
    if settings.model_backend == "transformers":
        return TransformersEngine(settings.base_model_path, settings.adapter_path)
    if settings.model_backend == "ollama":
        return OllamaEngine(settings.ollama_host, settings.ollama_finetuned_model, settings.ollama_base_model)
    return MockEngine()
