"""
Build the resume <-> job-description relevance dataset.

Pipeline:
  1. Read real resumes from the Kaggle "Resume Dataset" CSV
     (https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).
  2. For each resume, use Gemini as a synthetic-JD generator to produce
     one "matching" JD (same field, plausible fit) and one "mismatched"
     JD (different field / seniority), so the label distribution isn't
     bunched at the top end.
  3. Use Gemini again as a labeling oracle: given (resume, JD), produce
     a 0-100 relevance score + 2-sentence rationale.
  4. Emit instruction-tuning examples and split 80/10/10 train/val/test.

This is a synthetic-label dataset. Documented explicitly (see README) as
a limitation: labels reflect Gemini's judgment, not verified hiring
outcomes. Good enough to prove a fine-tune can specialize a small model
toward this scoring behavior; not a substitute for real recruiter labels.

Usage:
    export GEMINI_API_KEY=...   # free tier: https://aistudio.google.com/apikey
    python prepare_dataset.py --input data/raw/Resume.csv --n 1500

Resumable: progress is checkpointed to --checkpoint after every example,
so a killed/rate-limited run can just be restarted.
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

import pandas as pd
from google import genai
from google.genai.errors import APIError
from tqdm import tqdm

# gemini-3.6-flash's free tier is capped at 20 requests/day -- unusable for
# a few thousand calls. gemini-3.5-flash-lite's free tier allows 15
# requests/minute AND 500 requests/day. JD-generation and grading are
# combined into a single call per example (see COMBINED_PROMPT below) so
# the daily cap buys 500 examples/day instead of 250.
MODEL = "gemini-3.5-flash-lite"
MAX_REQUESTS_PER_MINUTE = 14


def _seconds_until_pacific_midnight() -> float:
    """Google's free-tier daily quota resets at midnight Pacific time."""
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (reset - now).total_seconds()


class RateLimiter:
    """Caps calls to a rolling per-minute window, shared across all worker threads."""

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


rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)

COMBINED_PROMPT = """You are helping build a training dataset for a resume-screening model. Do
two things in a single response, playing job-description writer then recruiter grader.

Step 1: Given the resume below, write a realistic job description (150-300
words: a title line, then responsibilities and requirements).
Mode: {mode}
- "matching": a job this candidate would be a strong-to-decent fit for (same
  general field/seniority as the resume, but don't make it a perfect echo of
  the resume text -- phrase it the way a real job posting would be written).
- "mismatched": a job in a different field, seniority level, or requiring
  skills this candidate mostly lacks, so a real screener would rate it a
  weak match.
Do not mention the resume or the candidate in the job description text.

Step 2: As an experienced technical recruiter, score how well the resume
matches the job description you just wrote (0-100). Score strictly on
evidence in the resume: skills, years of relevant experience, domain
overlap, seniority match. Do not reward keyword stuffing.

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


def call_gemini(client: genai.Client, prompt: str) -> str:
    backoff = 5.0
    attempt = 0
    while True:
        rate_limiter.acquire()
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt)
            return (resp.text or "").strip()
        except APIError as e:
            if "PerDay" in str(e):
                wait_s = _seconds_until_pacific_midnight() + 300
                print(f"  [info] daily free-tier quota hit -- sleeping ~{wait_s / 3600:.1f}h until reset...",
                      file=sys.stderr, flush=True)
                time.sleep(wait_s)
                continue  # doesn't count against the attempt budget below
            if e.code in (429, 500, 503) and attempt < 6:
                time.sleep(backoff)
                backoff = min(backoff * 1.8, 60)
                attempt += 1
                continue
            raise
        except OSError:
            # transient network/DNS hiccup (e.g. brief connectivity loss)
            if attempt < 6:
                time.sleep(backoff)
                backoff = min(backoff * 1.8, 60)
                attempt += 1
                continue
            raise


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


def process_one(client: genai.Client, resume: str, mode: str) -> dict | None:
    raw = call_gemini(client, COMBINED_PROMPT.format(mode=mode, resume=resume))
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
    text_col = next((c for c in df.columns if c.lower() in ("resume_str", "resume", "resume_text")), None)
    if text_col is None:
        raise ValueError(f"Couldn't find a resume text column in {list(df.columns)}")
    texts = df[text_col].dropna().astype(str)
    texts = texts[texts.str.len() > 200]  # drop near-empty rows
    texts = texts.drop_duplicates()
    text_list = texts.tolist()
    random.Random(seed).shuffle(text_list)
    return text_list[:n]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/raw/Resume.csv"),
                     help="Kaggle resume CSV path")
    ap.add_argument("--n", type=int, default=900,
                     help="number of resumes to sample (each produces 2 examples: matching + mismatched)")
    ap.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--checkpoint", type=Path, default=Path("data/raw/generation_checkpoint.jsonl"))
    ap.add_argument("--workers", type=int, default=4,
                     help="concurrency is capped by the global rate limiter regardless of this value")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Set GEMINI_API_KEY before running this script (free tier: https://aistudio.google.com/apikey).")
    if not args.input.exists():
        sys.exit(
            f"Resume CSV not found at {args.input}.\n"
            "Download it first, e.g.:\n"
            "  kaggle datasets download -d snehaanbhawal/resume-dataset -p data/raw --unzip"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    resumes = load_resumes(args.input, args.n, args.seed)
    jobs = [(r, mode) for r in resumes for mode in ("matching", "mismatched")]

    done = load_checkpoint(args.checkpoint)
    done_keys = {(d["input"][:80], d["meta"]["mode"]) for d in done}
    print(f"Resuming: {len(done)} examples already generated.")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    ckpt_f = open(args.checkpoint, "a", encoding="utf-8")

    def task(job):
        resume, mode = job
        key = (f"Resume: {resume}"[:80], mode)
        if key in done_keys:
            return None
        try:
            return process_one(client, resume, mode)
        except Exception as e:
            print(f"  [warn] failed one example: {e}", file=sys.stderr)
            return None

    remaining = [j for j in jobs if (f"Resume: {j[0]}"[:80], j[1]) not in done_keys]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in tqdm(pool.map(task, remaining), total=len(remaining), desc="labeling"):
            if result is not None:
                ckpt_f.write(json.dumps(result) + "\n")
                ckpt_f.flush()
    ckpt_f.close()

    all_rows = load_checkpoint(args.checkpoint)
    random.Random(args.seed).shuffle(all_rows)
    n = len(all_rows)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": all_rows[:n_train],
        "val": all_rows[n_train:n_train + n_val],
        "test": all_rows[n_train + n_val:],
    }
    for name, rows in splits.items():
        out_path = args.out_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                clean = {k: row[k] for k in ("instruction", "input", "output")}
                f.write(json.dumps(clean) + "\n")
        print(f"{name}: {len(rows)} examples -> {out_path}")

    scores = [r["meta"]["score"] for r in all_rows]
    print(f"\nTotal: {n} examples. Score distribution: "
          f"min={min(scores)} max={max(scores)} mean={sum(scores)/n:.1f}")
    print("IMPORTANT: the held-out test.jsonl must not be touched again until eval time.")


if __name__ == "__main__":
    main()
