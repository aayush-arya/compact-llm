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


_STOPWORDS = frozenset(
    "a an the and or but for nor so yet of to in on at by with from as is are was were be been "
    "being this that these those it its we you they he she our your their my his her i will would "
    "can could should may might must have has had do does did not no if then than into over under "
    "about across after before between during without within will work experience role team years "
    "year strong plus etc via per".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")


def _keywords(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2 and t not in _STOPWORDS}


def _seeded_unit(*parts: str) -> float:
    """Deterministic pseudo-random float in [0, 1) from the given strings."""
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _fit_signal(resume: str, jd: str) -> tuple[float, list[str], int]:
    """A cheap, explainable proxy for resume<->JD fit.

    Returns (true_score in [0,100], up to 6 shared keywords, count of JD
    keywords the resume never mentions). Real scoring is what
    MODEL_BACKEND=transformers / ollama do; this just keeps the demo coherent
    without weights.
    """
    r, j = _keywords(resume), _keywords(jd)
    if not j:
        return 50.0, [], 0
    shared = r & j
    coverage = len(shared) / len(j)          # fraction of JD requirements the resume hits
    depth = min(1.0, len(resume) / 2200)     # longer resume ~= more evidence to match on
    true_score = max(0.0, min(100.0, 170 * coverage + 10 * depth + 3))
    ranked = sorted(shared, key=lambda w: (-len(w), w))[:6]
    return true_score, ranked, len(j - r)


class MockEngine(InferenceEngine):
    """Deterministic heuristic scorer so the full stack is demoable without a
    GPU or trained weights. It is not a real model -- it turns keyword overlap
    into a plausible 0-100 score. The fine-tuned path is modelled as the
    decisive, calibrated, faster one; the base path as a vaguer generalist that
    hedges toward the middle of the scale -- so /compare tells the same story
    the real benchmark should. Swap MODEL_BACKEND to `transformers` or `ollama`
    for genuine output before quoting any numbers."""

    _DISCLAIMER = "Heuristic mock output (no model weights loaded)."

    def _score(self, resume: str, jd: str, use_adapter: bool) -> tuple[int, str]:
        true_score, shared, missing = _fit_signal(resume, jd)

        if use_adapter:
            jitter = (_seeded_unit(resume, jd, "ft") - 0.5) * 6  # +/- 3
            score = int(round(max(0, min(100, true_score + jitter))))
            if shared:
                rationale = (
                    f"{self._DISCLAIMER} Resume matches the JD on {', '.join(shared[:4])}; "
                    f"{missing} required term(s) go unmatched, which caps the score."
                )
            else:
                rationale = (
                    f"{self._DISCLAIMER} Almost no overlap with the JD's required skills, "
                    "so this reads as a weak match."
                )
        else:
            # generalist: compress toward 50 (hedged) and add wider noise
            jitter = (_seeded_unit(resume, jd, "base") - 0.5) * 18  # +/- 9
            score = int(round(max(0, min(100, 50 + (true_score - 50) * 0.55 + jitter))))
            rationale = (
                f"{self._DISCLAIMER} Base model, zero-shot: broad-strokes read of resume/JD "
                "fit, not calibrated to the 0-100 scale the fine-tune learned -- it hedges "
                "toward the middle."
            )
        return score, rationale

    def generate(self, resume: str, jd: str, use_adapter: bool) -> GenerationResult:
        start = time.perf_counter()
        score, rationale = self._score(resume, jd, use_adapter)
        # the fine-tuned 4B model answers directly; the base path pays for a longer prompt
        base = 0.12 if use_adapter else 0.34
        time.sleep(base + _seeded_unit(resume, jd, "lat") * 0.06)
        latency_ms = (time.perf_counter() - start) * 1000
        return GenerationResult(score=score, rationale=rationale, latency_ms=latency_ms)

    def generate_stream(self, resume: str, jd: str, use_adapter: bool) -> Iterator[str]:
        score, rationale = self._score(resume, jd, use_adapter)
        for i, word in enumerate(f"Score: {score}\nRationale: {rationale}".split(" ")):
            time.sleep(0.02)
            yield word if i == 0 else " " + word


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
