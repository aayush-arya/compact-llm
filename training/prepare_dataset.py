"""
Build the resume <-> job-description relevance dataset.

Pipeline:
  1. Read real resumes from the Kaggle "Resume Dataset" CSV
     (https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).
  2. For each resume, ask a large LLM to write five job descriptions spanning
     the fit spectrum -- from a near-ideal match down to a clear mismatch -- so
     the label distribution covers the whole 0-100 range instead of clustering
     at the extremes (the failure mode of an earlier "matching vs mismatched"
     two-way design).
  3. In the same call, the LLM scores the (resume, JD) pair 0-100 with a short
     rationale, judging only on evidence in the resume.
  4. Emit instruction-tuning examples and split 80/10/10, stratified by score
     band so train / val / test each span the full range.

This is a synthetic-label dataset. Documented explicitly (see README) as a
limitation: labels reflect the oracle model's judgment, not verified hiring
outcomes. Good enough to prove a fine-tune can specialize a small model toward
this scoring behavior; not a substitute for real recruiter labels.

Provider: set one of CEREBRAS_API_KEY (recommended -- 1M free tokens/day),
GROQ_API_KEY, or GEMINI_API_KEY in training/.env. The script auto-selects in
that order; --provider forces one. All three have a free, no-card tier.

Usage:
    python prepare_dataset.py --input data/raw/Resume.csv --resumes 120
    # -> data/processed/{train,val,test}.jsonl  (~600 examples: 5 per resume)

Resumable: progress is checkpointed after every example, so a killed or
rate-limited run just needs to be restarted with the same arguments.
"""
import argparse
import collections
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from tqdm import tqdm

try:  # let the key live in training/.env instead of the shell environment
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

# Per-provider defaults. `rpm` sits under each free tier's requests/minute cap;
# the binding constraint is usually tokens/day (Cerebras ~1M, Groq ~100k,
# Gemini is request-based). One combined call per example (JD generation +
# grading) keeps each example at a single request. `url` empty => Gemini's
# native REST shape; otherwise an OpenAI-compatible chat-completions endpoint.
PROVIDERS = {
    "cerebras": {
        "model": "gpt-oss-120b",
        "rpm": 25,
        "url": "https://api.cerebras.ai/v1/chat/completions",
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "rpm": 14,
        "url": "https://api.groq.com/openai/v1/chat/completions",
    },
    "gemini": {"model": "gemini-3.5-flash-lite", "rpm": 14, "url": ""},
}
PROVIDER_PRIORITY = ("cerebras", "groq", "gemini")
PROVIDER_ENV = {p: f"{p.upper()}_API_KEY" for p in PROVIDERS}

# Cap resume text fed to the oracle: enough to write a JD and grade fit, small
# enough to stay well inside free-tier token/day limits (and Cerebras' 8k ctx).
MAX_RESUME_CHARS = 1600

# Five job-description "distances" from the resume. The grader still scores each
# pair independently on evidence (see COMBINED_PROMPT); the mode only steers how
# far the generated JD sits from the candidate's actual background, which is
# what spreads the score distribution across the range.
MODES: dict[str, str] = {
    "strong": (
        "a role this candidate is an excellent fit for: the same function and "
        "seniority as the resume, where the resume already demonstrates every "
        "major requirement."
    ),
    "solid": (
        "a role in the same field and seniority as the resume, but which asks "
        "for one or two specific skills, tools, or domains the resume does not "
        "clearly show -- a good but imperfect fit."
    ),
    "partial": (
        "a role adjacent to the resume -- a neighbouring specialisation, or one "
        "notch of seniority away -- so only part of the candidate's experience "
        "transfers."
    ),
    "weak": (
        "a role that shares only a broad industry or a few transferable skills "
        "with the resume: wrong function or wrong domain, the kind of posting a "
        "screener would rate a weak match."
    ),
    "mismatch": (
        "a role in a clearly different field that needs skills this candidate "
        "does not have, with little transferable overlap."
    ),
}

COMBINED_PROMPT = """You are building a training dataset for a resume-screening model.
Do two things in a single response.

Step 1 -- job description. Given the resume below, write a realistic job
description (150-300 words: a title line, then responsibilities and
requirements). Aim for this fit level relative to the candidate: {mode_guidance}
Do not mention the resume or the candidate. Write it the way a real posting
would read, not as an echo of the resume text.

Step 2 -- score. As an experienced technical recruiter, score how well the
resume matches the job description you just wrote (0-100). Judge ONLY on
evidence in the resume: relevant skills, years of relevant experience, domain
overlap, seniority match. Score what is actually on the page, not how the JD
was commissioned. Do not reward keyword stuffing.

Resume:
{resume}

Respond in EXACTLY this format, nothing else, no markdown:
Job Description: <the job description text>
Score: <integer 0-100>
Rationale: <exactly 2 sentences explaining the score, citing specific
resume/JD evidence>
"""

INSTRUCTION = (
    "Score how well this resume matches this job description (0-100) and "
    "explain why in 2 sentences."
)


def _seconds_until_pacific_midnight() -> float:
    """Google's free-tier daily quota resets at midnight Pacific time."""
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (reset - now).total_seconds()


class RateLimiter:
    """Caps calls to a rolling per-minute window, shared across worker threads."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.timestamps: collections.deque = collections.deque()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] > 60:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.max_per_minute:
                    self.timestamps.append(now)
                    return
                sleep_for = 60 - (now - self.timestamps[0]) + 0.1
            time.sleep(sleep_for)


class LLMClient:
    """One combined JD-generation + grading call, against Cerebras, Groq or
    Gemini.

    Cerebras and Groq are OpenAI-compatible chat endpoints; Gemini uses its
    native generateContent shape. All reached over plain HTTP -- Gemini's own
    SDK trips over its new `AQ.`-prefixed keys, so this picks the auth header by
    key format instead. Retries 429/5xx with backoff; for Gemini's hard daily
    cap it sleeps until the Pacific-midnight reset.
    """

    def __init__(self, provider: str, api_key: str, model: str, rpm: int, url: str):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.url = url
        self.limiter = RateLimiter(rpm)
        self.http = httpx.Client(timeout=120)

    def _request(self, prompt: str) -> httpx.Response:
        if self.url:  # OpenAI-compatible (Cerebras, Groq)
            return self.http.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                },
            )
        # gemini: AQ. auth keys want a Bearer header; classic AIza keys use x-goog-api-key
        header = (
            {"Authorization": f"Bearer {self.api_key}"}
            if self.api_key.startswith("AQ.")
            else {"x-goog-api-key": self.api_key}
        )
        return self.http.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            headers=header,
            json={"contents": [{"parts": [{"text": prompt}]}]},
        )

    def _extract(self, data: dict) -> str:
        if self.url:
            return data["choices"][0]["message"]["content"].strip()
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()

    def generate(self, prompt: str) -> str:
        backoff = 5.0
        attempt = 0
        while True:
            self.limiter.acquire()
            try:
                resp = self._request(prompt)
            except (httpx.TransportError, OSError):
                if attempt < 6:
                    time.sleep(backoff)
                    backoff = min(backoff * 1.8, 60)
                    attempt += 1
                    continue
                raise

            if resp.status_code == 200:
                return self._extract(resp.json())

            body = resp.text
            lower = body.lower()
            hit_daily = (
                (self.provider == "gemini" and "PerDay" in body)
                or (resp.status_code == 429 and ("per day" in lower or "daily" in lower or "tokens per day" in lower))
            )
            if hit_daily:
                wait_s = (
                    _seconds_until_pacific_midnight() + 300
                    if self.provider == "gemini"
                    else 3600  # OpenAI-compat daily limits recover on a rolling window
                )
                print(
                    f"  [info] daily free-tier quota hit -- sleeping ~{wait_s / 3600:.1f}h...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait_s)
                continue  # doesn't count against the attempt budget
            if resp.status_code in (429, 500, 502, 503) and attempt < 6:
                retry_after = float(resp.headers.get("retry-after", 0) or 0)
                time.sleep(max(retry_after, backoff))
                backoff = min(backoff * 1.8, 60)
                attempt += 1
                continue
            raise RuntimeError(f"{self.provider} {resp.status_code}: {body[:200]}")


def parse_combined(text: str) -> tuple[str, int, str] | None:
    jd_m = re.search(r"Job Description:\s*(.+?)\s*Score:", text, re.DOTALL)
    score_m = re.search(r"Score:\s*(\d{1,3})", text)
    rationale_m = re.search(r"Rationale:\s*(.+)", text, re.DOTALL)
    if not jd_m or not score_m or not rationale_m:
        return None
    jd = jd_m.group(1).strip()
    score = max(0, min(100, int(score_m.group(1))))
    rationale = " ".join(rationale_m.group(1).strip().split())
    return jd, score, rationale


def process_one(llm: LLMClient, resume: str, mode: str) -> dict | None:
    raw = llm.generate(COMBINED_PROMPT.format(mode_guidance=MODES[mode], resume=resume))
    parsed = parse_combined(raw)
    if parsed is None:
        return None
    jd, score, rationale = parsed
    return {
        "instruction": INSTRUCTION,
        "input": f"Resume: {resume}\n\nJob Description: {jd}",
        "output": f"Score: {score}\nRationale: {rationale}",
        "meta": {"mode": mode, "score": score},
    }


def load_resumes(csv_path: Path, n: int, seed: int) -> list[str]:
    df = pd.read_csv(csv_path)
    text_col = next(
        (c for c in df.columns if c.lower() in ("resume_str", "resume", "resume_text")), None
    )
    if text_col is None:
        raise ValueError(f"Couldn't find a resume text column in {list(df.columns)}")
    texts = df[text_col].dropna().astype(str)
    texts = texts[texts.str.len() > 200]  # drop near-empty rows
    texts = texts.drop_duplicates()
    text_list = texts.tolist()
    random.Random(seed).shuffle(text_list)
    # Truncate once, at the source, so the oracle grades exactly the text that
    # ends up in the training example -- and so token usage stays predictable.
    return [
        (re.sub(r"\s+\S*$", " …", t[:MAX_RESUME_CHARS]) if len(t) > MAX_RESUME_CHARS else t)
        for t in text_list[:n]
    ]


def load_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_split(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """Split 80/10/10 within each 20-point score band, so train / val / test all
    span the range instead of a random slice landing lopsided."""
    rng = random.Random(seed)
    buckets: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        band = min(4, r["meta"]["score"] // 20)  # 0-19, 20-39, 40-59, 60-79, 80-100
        buckets[band].append(r)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for band in sorted(buckets):
        items = buckets[band]
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train += items[:n_train]
        val += items[n_train : n_train + n_val]
        test += items[n_train + n_val :]

    for split in (train, val, test):
        rng.shuffle(split)
    return {"train": train, "val": val, "test": test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", type=Path, default=Path("data/raw/Resume.csv"), help="Kaggle resume CSV path"
    )
    ap.add_argument(
        "--resumes",
        type=int,
        default=120,
        help="number of resumes to sample (each produces 5 examples, one per fit mode)",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    ap.add_argument(
        "--checkpoint", type=Path, default=Path("data/raw/dataset_checkpoint.jsonl")
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="concurrency is capped by the global rate limiter regardless of this value",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--provider",
        choices=["auto", *PROVIDER_PRIORITY],
        default="auto",
        help="labeling oracle; 'auto' uses whichever of "
        + " / ".join(PROVIDER_ENV.values())
        + " is set (in that order)",
    )
    ap.add_argument("--model", default=None, help="override the provider's default model")
    ap.add_argument(
        "--list-models",
        action="store_true",
        help="print the models the selected provider's key can use, then exit",
    )
    args = ap.parse_args()

    keys = {p: os.environ.get(env) for p, env in PROVIDER_ENV.items()}
    if args.provider == "auto":
        provider = next((p for p in PROVIDER_PRIORITY if keys.get(p)), None)
    else:
        provider = args.provider
    api_key = keys.get(provider) if provider else None
    if not provider or not api_key:
        sys.exit(
            "No API key found. Put ONE of these in training/.env (git-ignored):\n"
            "  CEREBRAS_API_KEY=...  free, 1M tokens/day: https://cloud.cerebras.ai\n"
            "  GROQ_API_KEY=...      free, instant:       https://console.groq.com/keys\n"
            "  GEMINI_API_KEY=...    free tier:           https://aistudio.google.com/apikey"
        )
    cfg = PROVIDERS[provider]

    if args.list_models:
        if not cfg["url"]:
            base = "https://generativelanguage.googleapis.com/v1beta/models"
            hdr = (
                {"Authorization": f"Bearer {api_key}"}
                if api_key.startswith("AQ.")
                else {"x-goog-api-key": api_key}
            )
        else:
            base = cfg["url"].rsplit("/", 2)[0] + "/models"  # .../v1/chat/completions -> .../v1/models
            hdr = {"Authorization": f"Bearer {api_key}"}
        r = httpx.get(base, headers=hdr, timeout=30)
        r.raise_for_status()
        payload = r.json()
        rows = payload.get("data") or payload.get("models") or []
        print("models:")
        for m in rows:
            print("  " + (m.get("id") or m.get("name")))
        limits = {k: v for k, v in r.headers.items() if "ratelimit" in k.lower()}
        if limits:
            print("rate-limit headers:")
            for k, v in sorted(limits.items()):
                print(f"  {k}: {v}")
        return

    if not args.input.exists():
        sys.exit(
            f"Resume CSV not found at {args.input}.\n"
            "Download it first, e.g.:\n"
            "  kaggle datasets download -d snehaanbhawal/resume-dataset -p data/raw --unzip"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    resumes = load_resumes(args.input, args.resumes, args.seed)
    jobs = [(r, mode) for r in resumes for mode in MODES]

    done = load_checkpoint(args.checkpoint)
    done_keys = {(d["input"][:80], d["meta"]["mode"]) for d in done}
    print(f"Resuming: {len(done)} examples already generated.")

    model = args.model or cfg["model"]
    llm = LLMClient(provider, api_key, model, cfg["rpm"], cfg["url"])
    print(f"Provider: {provider} · model: {model}")
    ckpt_f = open(args.checkpoint, "a", encoding="utf-8")

    def task(job):
        resume, mode = job
        key = (f"Resume: {resume}"[:80], mode)
        if key in done_keys:
            return None
        try:
            return process_one(llm, resume, mode)
        except Exception as e:  # noqa: BLE001 - one bad example shouldn't kill the run
            print(f"  [warn] failed one example ({mode}): {e}", file=sys.stderr)
            return None

    remaining = [j for j in jobs if (f"Resume: {j[0]}"[:80], j[1]) not in done_keys]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in tqdm(pool.map(task, remaining), total=len(remaining), desc="labeling"):
            if result is not None:
                ckpt_f.write(json.dumps(result) + "\n")
                ckpt_f.flush()
    ckpt_f.close()

    all_rows = load_checkpoint(args.checkpoint)
    if not all_rows:
        sys.exit(
            "No examples generated -- nothing to split. Check the errors above "
            "(a 401 means the API key in training/.env is wrong or the wrong type)."
        )
    splits = stratified_split(all_rows, args.seed)
    for name, rows in splits.items():
        out_path = args.out_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                clean = {k: row[k] for k in ("instruction", "input", "output")}
                f.write(json.dumps(clean) + "\n")
        print(f"{name}: {len(rows)} examples -> {out_path}")

    scores = [r["meta"]["score"] for r in all_rows]
    by_band = collections.Counter(min(4, s // 20) for s in scores)
    band_labels = ["0-19", "20-39", "40-59", "60-79", "80-100"]
    dist = "  ".join(f"{band_labels[b]}:{by_band.get(b, 0)}" for b in range(5))
    print(
        f"\nTotal: {len(all_rows)} examples. "
        f"Score min={min(scores)} max={max(scores)} mean={sum(scores) / len(scores):.1f}"
    )
    print(f"Distribution by band: {dist}")
    print("IMPORTANT: the held-out test.jsonl must not be touched again until eval time.")


if __name__ == "__main__":
    main()
